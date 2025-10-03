import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

# --------- 基元：多头注意力 + 残差/归一化 ----------
class MAB(nn.Module):
  
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.fc_o = nn.Linear(d_model, d_model, bias=False)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(4 * d_model, d_model, bias=True)
        )
        self.attn_drop = nn.Dropout(dropout)
        self.ff_drop = nn.Dropout(dropout)

    def forward(self, Q, K, attn_mask: Optional[torch.Tensor] = None):
        """
        attn_mask: [B, N, M]，True 表示屏蔽（不参与），可选
        """
        B, N, _ = Q.shape
        M = K.size(1)

        q = self.W_q(Q).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, N, d_head]
        k = self.W_k(K).view(B, M, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, M, d_head]
        v = self.W_v(K).view(B, M, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, M, d_head]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)     # [B, H, N, M]

        if attn_mask is not None:
            # True -> -inf（被屏蔽）
            attn = attn.masked_fill(attn_mask.unsqueeze(1), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)                                            # [B, H, N, d_head]
        out = out.transpose(1, 2).contiguous().view(B, N, -1)                  # [B, N, d_model]
        out = self.fc_o(out)

        # 残差 + LN
        y = self.ln1(Q + out)
        y2 = self.ff_drop(self.ff(y))
        y = self.ln2(y + y2)
        return y


class SAB(nn.Module):
    """ Self-Attention Block: 直接用 MAB(Q=K=X) """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mab = MAB(d_model, n_heads, dropout)

    def forward(self, X, attn_mask: Optional[torch.Tensor] = None):
        return self.mab(X, X, attn_mask=attn_mask)


class ISAB(nn.Module):
    """
    Induced Set Attention Block
    用可学习的诱导点 I (num_inducing × d_model) 降维集合交互
    """
    def __init__(self, d_model: int, n_heads: int, num_inducing: int, dropout: float = 0.1):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, d_model) * 0.02)
        self.mab1 = MAB(d_model, n_heads, dropout)  # H = MAB(I, X)
        self.mab2 = MAB(d_model, n_heads, dropout)  # Y = MAB(X, H)

    def forward(self, X, attn_mask: Optional[torch.Tensor] = None):
        B = X.size(0)
        I = self.I.expand(B, -1, -1)                 # [B, m, d]
        H = self.mab1(I, X, attn_mask=None)          # 诱导点先从 X 聚合信息
        Y = self.mab2(X, H, attn_mask=None)          # 再让 X 从诱导点回读
        return Y


class PMA(nn.Module):
    """
    Pooling by Multihead Attention
    用 k 个可学习的种子向量 S 来池化集合 -> [B, k, d_model]
    """
    def __init__(self, d_model: int, n_heads: int, k: int = 1, dropout: float = 0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, d_model) * 0.02)
        self.mab = MAB(d_model, n_heads, dropout)

    def forward(self, X):
        B = X.size(0)
        S = self.S.expand(B, -1, -1)   # [B, k, d_model]
        return self.mab(S, X, attn_mask=None)  # [B, k, d_model]


# --------- 输入侧：multi-hot / indices -> 元素级 embedding ---------
class MultiHotToSet(nn.Module):
    """
    将 multi-hot/BOW 或 稀疏 indices 转为元素嵌入序列:
    - multi-hot: Xmh [B, V] -> 选出非零 token 的 embedding，长度不等（使用阈值/采样）
    - indices: List[List[int]] 或稀疏张量 -> 直接 lookup embedding
    为了高效也支持 dense 线性投影: Xmh @ E
    """
    def __init__(self, vocab_size: int, d_model: int, share_embedding: Optional[nn.Embedding] = None,
                 use_dense_matmul: bool = True):
        super().__init__()
        self.use_dense_matmul = use_dense_matmul
        if share_embedding is None:
            self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        else:
            self.emb = share_embedding  # 可与 token encoder 共享
        # 注意：dense matmul 路径会产生一个“汇聚后”的 [B, d_model]，不展开为序列

    def from_multi_hot(self, Xmh: torch.Tensor, expand_as_sequence: bool = False, max_len: Optional[int] = None):
        """
        Xmh: [B, V] (0/1 或 计数/权重)
        expand_as_sequence=False:
            返回 bag 向量: [B, d_model] = Xmh @ E
        expand_as_sequence=True:
            返回元素级序列（会把 1 的 index 展开并 lookup embedding），
            形状拼接成 [B, L, d_model]（需要 pad 到 max_len）
        """
        B, V = Xmh.shape
        if not expand_as_sequence or self.use_dense_matmul:
            bag = Xmh.float() @ self.emb.weight  # [B, d_model]
            return bag, None  # None 表示无长度张量
        else:
            # 稀疏展开为序列（L 可变），并 pad 到 max_len
            device = Xmh.device
            idxs = Xmh.nonzero(as_tuple=False)  # [N, 2], 每行 [b, v]
            # 分桶
            lists: List[List[int]] = [[] for _ in range(B)]
            for b, v in idxs.tolist():
                lists[b].append(v)
            if max_len is None:
                max_len = max(1, max(len(lst) for lst in lists))
            # pad & lookup
            seq = torch.zeros(B, max_len, dtype=torch.long, device=device)
            lengths = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                take = lists[b][:max_len]
                L = len(take)
                lengths[b] = L
                if L > 0:
                    seq[b, :L] = torch.tensor(take, dtype=torch.long, device=device)
            X = self.emb(seq)  # [B, L, d_model]
            return X, lengths

    def from_indices(self, indices: List[List[int]], device, max_len: Optional[int] = None):
        B = len(indices)
        if max_len is None:
            max_len = max(1, max(len(lst) for lst in indices))
        seq = torch.zeros(B, max_len, dtype=torch.long, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            take = indices[b][:max_len]
            L = len(take)
            lengths[b] = L
            if L > 0:
                seq[b, :L] = torch.tensor(take, dtype=torch.long, device=device)
        X = self.emb(seq)  # [B, L, d_model]
        return X, lengths


# --------- Set Transformer Encoder（面向集合，Permutation-invariant） ----------
class SetTransformerEncoder(nn.Module):
    """
    用法 1（multi-hot/BOW，推荐）:
        enc = SetTransformerEncoder(vocab_size=V, d_model=512, d_latent=64, n_heads=8,
                                    num_inducing=32, num_sab=1, dropout=0.1)
        z = enc(multi_hot=Xmh)  # Xmh: [B, V] -> 输出 [B, 1, d_latent]

    用法 2（indices 列表）:
        z = enc(indices=token_lists)   # List[List[int]]，可变长

    说明：
        - multi-hot 默认用 dense matmul 得到 bag 向量，再升维成长度 1 的“集合”（等价 |S|=1）。
        - 若想显式展开为元素级序列，可传 expand_as_sequence=True（更贵，能建模元素间交互）。
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 d_latent: int = 64,
                 n_heads: int = 8,
                 num_inducing: int = 32,
                 num_sab: int = 1,
                 k_pma: int = 1,
                 dropout: float = 0.1,
                 share_embedding: Optional[nn.Embedding] = None,
                 use_dense_matmul: bool = True,
                 expand_as_sequence: bool = False,
                 max_len: Optional[int] = None):
        super().__init__()

        self.expand_as_sequence = expand_as_sequence
        self.max_len = max_len

        self.input_builder = MultiHotToSet(
            vocab_size=vocab_size,
            d_model=d_model,
            share_embedding=share_embedding,
            use_dense_matmul=use_dense_matmul
        )

        # 如果展开成序列 -> ISAB/SAB；否则 bag 向量 -> 当作 |S|=1 的集合进入 SAB/ISAB 也可工作
        self.isab = ISAB(d_model, n_heads, num_inducing, dropout)
        self.sabs = nn.ModuleList([SAB(d_model, n_heads, dropout) for _ in range(num_sab)])
        self.pma = PMA(d_model, n_heads, k=k_pma, dropout=dropout)  # k=1 输出 [B,1,d_model]

        self.out = nn.Linear(d_model, d_latent, bias=False)

    def forward(self,
                multi_hot: Optional[torch.Tensor] = None,
                indices: Optional[List[List[int]]] = None):
        """
        二选一提供：
          - multi_hot: [B, V]（0/1/权重均可）
          - indices: List[List[int]]（每个样本一个 token id 列表）
        """
        assert (multi_hot is None) ^ (indices is None), "multi_hot 和 indices 必须二选一提供"

        # 1) 构建元素级序列 X: [B, L, d_model] 和 lengths（可选）
        if multi_hot is not None:
            if self.expand_as_sequence:
                X, lengths = self.input_builder.from_multi_hot(
                    multi_hot, expand_as_sequence=True, max_len=self.max_len
                )  # [B, L, d_model]
                if lengths is None:
                    lengths = torch.full((multi_hot.size(0),), fill_value=X.size(1),
                                         dtype=torch.long, device=X.device)
            else:
                # bag: [B, d_model] -> 视作长度 1 的集合 [B, 1, d_model]
                bag, _ = self.input_builder.from_multi_hot(multi_hot, expand_as_sequence=False)
                X = bag.unsqueeze(1)        # [B, 1, d_model]
                lengths = torch.ones(bag.size(0), dtype=torch.long, device=bag.device)
        else:
            # indices 列表
            device = self.input_builder.emb.weight.device
            X, lengths = self.input_builder.from_indices(indices, device=device, max_len=self.max_len)  # [B, L, d_model]

        # 2) 可选：根据 lengths 生成 attention mask（SAB/ISAB 可不用 mask；若想严格屏蔽 padding，就做 mask）
        B, L, _ = X.shape
        pad_mask = None
        if lengths is not None:
            ar = torch.arange(L, device=X.device).unsqueeze(0).expand(B, L)  # [B,L]
            valid = ar < lengths.unsqueeze(1)                                 # True on valid
            pad_mask = ~valid                                                 # True -> padding
            # SAB/ISAB 的 MAB 接收 [B, N, M] 的 mask（True=masked），这里简化为 None（通常没问题）
            # 若你想严格屏蔽，把 pad_mask 转成 [B,N,M] 的形式传入 MAB 即可。

        # 3) Set Transformer 主干：ISAB -> (SAB...) -> PMA
        Y = self.isab(X, attn_mask=None)  # 通常不需要 mask；如需可构造成 [B,N,M] 的布尔张量
        for sab in self.sabs:
            Y = sab(Y, attn_mask=None)

        pooled = self.pma(Y)              # [B, k, d_model]，k=1 -> [B,1,d_model]
        z = self.out(pooled)              # [B, 1, d_latent]
        return z
