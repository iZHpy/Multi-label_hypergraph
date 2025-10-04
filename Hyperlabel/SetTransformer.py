import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from Hyperlabel.SubLayers import MultiHeadAttention


class SAB(nn.Module):
    """ Self-Attention Block: 直接用 MAB(Q=K=X) """
    def __init__(self, d_model: int, n_heads: int, d_k : int, d_v: int, dropout: float = 0.1):
        super().__init__()
        self.mab = MultiHeadAttention(n_heads, d_model, d_k=d_k, d_v=d_v, dropout=dropout)

    def forward(self, X, attn_mask: Optional[torch.Tensor] = None):
        out, _ = self.mab(X, X, X, attn_mask=attn_mask)
        return out


class ISAB(nn.Module):
    """
    Induced Set Attention Block
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, num_inducing: int, dropout: float = 0.1):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, d_model) * 0.02)
        self.mab1 = MultiHeadAttention(n_heads, d_model, d_k=d_k, d_v=d_v, dropout=dropout)  # H = MAB(I, X)
        self.mab2 = MultiHeadAttention(n_heads, d_model, d_k=d_k, d_v=d_v, dropout=dropout)  # Y = MAB(X, H)

    def forward(self, X, attn_mask: Optional[torch.Tensor] = None):
        B = X.size(0)
        I = self.I.expand(B, -1, -1)                 # [B, m, d]
        H, _ = self.mab1(I, X, X, attn_mask=None)          # 诱导点先从 X 聚合信息
        Y, _ = self.mab2(X, H, H, attn_mask=None)          # 再让 X 从诱导点回读
        return Y


class PMA(nn.Module):
    """
    Pooling by Multihead Attention
    """
    def __init__(self, d_model: int, n_heads: int, d_k: int, d_v: int, k: int = 1, dropout: float = 0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, k, d_model) * 0.02)
        self.mab = MultiHeadAttention(n_heads, d_model, d_k=d_k, d_v=d_v, dropout=dropout)

    def forward(self, X):
        B = X.size(0)
        S = self.S.expand(B, -1, -1)   # [B, k, d_model]
        out, _ = self.mab(S, X, X, attn_mask=None)  # [B, k, d_model]
        return out


class MultiHotToSet(nn.Module):
    """
    multi-hot: Xmh [B, V] ->  dense: Xmh @ E
    """
    def __init__(self, vocab_size: int, d_model: int, use_dense_matmul: bool = True):
        super().__init__()
        self.use_dense_matmul = use_dense_matmul

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 注意：dense matmul 路径会产生一个“汇聚后”的 [B, d_model]，不展开为序列

    def from_multi_hot(self, Xmh: torch.Tensor, expand_as_sequence: bool = False, max_len: Optional[int] = None):
        """
        Xmh: [B, V]
        expand_as_sequence=False:
            bag : [B, d_model] = Xmh @ E
        """
        B, V = Xmh.shape
        if not expand_as_sequence or self.use_dense_matmul:
            bag = Xmh.float() @ self.emb.weight  # [B, d_model]
            return bag, None 
        else:
            device = Xmh.device
            idxs = Xmh.nonzero(as_tuple=False) 
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


class SetTransformerEncoder(nn.Module):
    """
    input: multi_hot [B, V]
    output: z [B, 1, d_latent]
    1) multi_hot -> embedding -> set of elements [B, L, d_model]
    2) ISAB -> (SAB...) -> PMA -> [B, 1, d_model]
    3) Linear -> [B, 1, d_latent]
    4) return z [B, 1, d_latent]
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 d_latent: int = 64,
                 n_heads: int = 8,
                 d_k: int = 64,
                 d_v: int = 64,
                 num_inducing: int = 32,
                 num_sab: int = 1,
                 k_pma: int = 1,
                 dropout: float = 0.1,
                 use_dense_matmul: bool = True,
                 max_len: Optional[int] = None):
        super().__init__()
        
        self.max_len = max_len
        self.layers = layers

        self.input_builder = MultiHotToSet(
            vocab_size=vocab_size,
            d_model=d_model,
            use_dense_matmul=use_dense_matmul
        )

        # 如果展开成序列 -> ISAB/SAB；否则 bag 向量 -> 当作 |S|=1 的集合进入 SAB/ISAB 也可工作
        self.isab = ISAB(d_model, n_heads, d_k, d_v, num_inducing, dropout=dropout)
        self.sabs = nn.ModuleList([SAB(d_model, n_heads, d_k, d_v, dropout) for _ in range(num_sab)])
        self.pma = PMA(d_model, n_heads, d_k, d_v, k=k_pma, dropout=dropout)  # k=1 输出 [B,1,d_model]

        self.out = nn.Linear(d_model, d_latent, bias=False)

    def forward(self,
                multi_hot: Optional[torch.Tensor] = None):
        """
        二选一提供：
          - multi_hot: [B, V]
        """


        # 1) 构建元素级序列 X: [B, L, d_model] 和 lengths（可选）
        if multi_hot is not None:
            # bag: [B, d_model] -> 视作长度 1 的集合 [B, 1, d_model]
            bag, _ = self.input_builder.from_multi_hot(multi_hot, expand_as_sequence=False)
            X = bag.unsqueeze(1)        # [B, 1, d_model]
            lengths = torch.ones(bag.size(0), dtype=torch.long, device=bag.device)

        # 3) Set Transformer 主干：ISAB -> (SAB...) -> PMA
        Y = self.isab(X, attn_mask=None)  # 通常不需要 mask；如需可构造成 [B,N,M] 的布尔张量
        for sab in self.sabs:
            Y = sab(Y, attn_mask=None)

        pooled = self.pma(Y)              # [B, k, d_model]，k=1 -> [B,1,d_model]
        z = self.out(pooled)              # [B, 1, d_latent]
        return z
