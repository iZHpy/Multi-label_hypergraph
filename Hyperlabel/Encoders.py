import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Hyperlabel.Constants as Constants
from Hyperlabel.Layers import EncoderLayer,DecoderLayer
from Hyperlabel.SubLayers import ScaledDotProductAttention
from Hyperlabel.SubLayers import PositionwiseFeedForward
from Hyperlabel.SubLayers import XavierLinear
from pdb import set_trace as stop 
from Hyperlabel import utils
import copy



class GraphEncoder(nn.Module):
    def __init__(
            self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_latent=64, d_inner_hid=1024, feat_mode='tokens', enc_transform='special_token',
            special_token_init='normal', dropout=0.1, no_enc_pos_embedding=False):

        super(GraphEncoder, self).__init__()

        n_position = n_max_seq + 1  # 0, 1, 2, ..., N
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.latent_dim = d_latent
        self.feat_mode = feat_mode
        self.enc_transform = enc_transform
        self.dropout = nn.Dropout(dropout)

        if feat_mode == 'tokens':
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        else:
            raise ValueError("Use MLPEncoder/DeepSets/SetTransformer for float features")
            
        if no_enc_pos_embedding is False:
            self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
            self.position_enc.weight.data = utils.position_encoding_init(n_position, d_word_vec)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        if enc_transform == 'special_token':
            if special_token_init == 'random':
                self.special_token_emb = nn.Parameter(torch.randn(1, 1, d_model))
            elif special_token_init == 'xavier_uniform':  # used with linear activation or tanh
                self.special_token_emb = nn.Parameter(torch.Tensor(1, 1, d_model))
                nn.init.xavier_uniform_(self.special_token_emb)
            elif special_token_init == 'xavier_normal':  # used with linear activation or tanh
                self.special_token_emb = nn.Parameter(torch.Tensor(1, 1, d_model))
                nn.init.xavier_normal_(self.special_token_emb)
            elif special_token_init == 'kaiming_uniform':  # used in ReLU based deep networks
                self.special_token_emb = nn.Parameter(torch.Tensor(1, 1, d_model))
                nn.init.kaiming_uniform_(self.special_token_emb, mode='fan_in', nonlinearity='relu')
            elif special_token_init == 'kaiming_normal':  # used in ReLU based deep networks
                self.special_token_emb = nn.Parameter(torch.Tensor(1, 1, d_model))
                nn.init.kaiming_normal_(self.special_token_emb, mode='fan_out', nonlinearity='relu')
            elif special_token_init == 'normal':  # used in BERT
                self.special_token_emb = nn.Parameter(torch.normal(0, 0.02, size=(1, 1, d_model)))
            else:
                raise ValueError("Unsupported initialization method")
        self.encoder_project = nn.Linear(d_model, d_latent)
        
    def forward(self, src_seq, adj, src_pos):
        batch_size = src_seq.size(0)
        enc_input = self.src_word_emb(src_seq)
        pad_idx = Constants.PAD
        num_pos = self.position_enc.num_embeddings  # 应该等于 n_max_seq + 1

        # 如果有负数/越界，先打印出来定位
        min_v = int(src_pos.min())
        max_v = int(src_pos.max())
        assert min_v >= 0 and max_v < num_pos, \
                    f"src_pos out of range: min={min_v}, max={max_v}, allowed=[0, {num_pos-1}] (PAD={pad_idx})"
        if hasattr(self, 'position_enc'):
            enc_input += self.position_enc(src_pos)
        
        if self.enc_transform == 'special_token':
            special_token = self.special_token_emb.expand(batch_size, -1, -1)
            enc_input = torch.cat([special_token, enc_input], dim=1)
            src_seq = F.pad(src_seq, (1, 0),
                            value=-1)  # Pad src_seq with an extra token, set it to be -1, not the same as the padding

        enc_output = enc_input
        enc_slf_attn_mask = utils.get_attn_padding_mask(src_seq, src_seq)  # this mask is used to remove the affect of padding from self-attention

        if adj:
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.float32)
            for idx in range(len(adj)):
                if self.enc_transform == 'special_token':
                    adj_size = adj[idx].size(0)
                    if self.enc_transform == 'special_token':
                        enc_slf_attn_mask[idx][1:adj_size + 1, 1:adj_size + 1] = utils.swap_0_1(adj[idx], 1, 0)
                    else:
                        enc_slf_attn_mask[idx][:adj_size, :adj_size] = utils.swap_0_1(adj[idx], 1, 0)
            enc_slf_attn_mask = enc_slf_attn_mask.type(torch.uint8)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)

        if self.enc_transform == 'max':
            enc_output = F.max_pool1d(enc_output.transpose(1, 2), enc_output.size(1)).squeeze()
        elif self.enc_transform == 'mean':
            enc_output = enc_output.sum(1) / ((src_seq > 0).sum(dim=1).float().view(-1, 1))
        elif self.enc_transform == 'special_token':
            enc_output = enc_output[:, 0, :]  # Take the first token (special token) as the output
        else:
            raise ValueError("not use enc_transform")

        enc_output = enc_output.view(batch_size, 1, -1)
        enc_output = self.encoder_project(enc_output)

        return enc_output

   
class ResidualMLP(nn.Module):
    ''' A residual connection followed by a layer norm '''
    def __init__(self, in_dim, d_hidden, dropout):
        super(ResidualMLP, self).__init__()
        self.layers = [nn.Linear(in_dim, d_hidden),
                       nn.ReLU(),
                       nn.Dropout(dropout),
                       nn.LayerNorm(d_hidden),
                       nn.Linear(d_hidden, in_dim)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x) + x

class MLPEncoder(nn.Module):
    """
    input: multi-hot vector [B, V], V is vocab size
    output: z [B, 1, d_latent]
    """
    def __init__(self, d_in, d_model=512, d_hidden=512, d_latent=64, n_layers=3, dropout=0.1, pool="mean"):
        super().__init__()

        self.pool = pool
        self.emb = nn.Linear(d_in, d_model, bias=False)
        layers = []
        in_dim = d_model
        self.blocks = nn.ModuleList([ResidualMLP(in_dim, d_hidden, dropout) 
                                     for _ in range(n_layers)])
        self.latent_proj = nn.Linear(in_dim, d_latent)
    def forward(self, multi_hot):
        # multi_hot: [B, V] -> bag embedding = multi_hot @ E (E=[V,d_model])
        out = self.emb(multi_hot)  # [B, d_model]

        if self.pool == "mean":
            counts = multi_hot.sum(-1, keepdim=True).clamp_min(1.0)
            out = out / counts
        for block in self.blocks:
            out = block(out)
        z = self.latent_proj(out)
        z = z.unsqueeze(1)  # [B, 1, d_latent]
        return z

    
class DeepSetsEncoder(nn.Module):
    """
    输入:
      - multi_hot: [B, V] 

    输出:
      - z: [B, 1, d_latent]   
    """
    def __init__(self, vocab_size, d_model=512, d_latent=64, dropout=0.1,  pool="mean"):
        super().__init__()
        self.pool = pool
        # embedding embedding E
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # aggregate func φ
        self.phi = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # map to latent space ρ
        self.rho = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_latent)
        )

    def forward(self, multi_hot: torch.Tensor):
        """
        multi_hot: [B, V]  
        """
        # [B, V] @ [V, d_model] = [B, d_model] (bag embedding)
        emb = multi_hot.float() @ self.emb.weight  # sum pooling by default

        if self.pool == "mean":
            counts = multi_hot.sum(-1, keepdim=True).clamp_min(1.0)
            emb = emb / counts
        elif self.pool == "max":
            raise NotImplementedError("max pooling not implemented yet")

        h = self.phi(emb)
        z = self.rho(h).unsqueeze(1)  # [B,1,d_latent]
        return z
