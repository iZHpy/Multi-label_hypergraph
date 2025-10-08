import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Hyperlabel import utils
from Hyperlabel.Layers import DecoderLayer
from Hyperlabel.SubLayers import XavierLinear

class ComplexDecoder(nn.Module):
    def __init__(self, feature_dim, num_labels, hidden_dim=512):
        super(ComplexDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.num_labels = num_labels

        # Attention mechanism
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        self.key_proj = nn.Linear(feature_dim, hidden_dim)
        self.value_proj = nn.Linear(feature_dim, hidden_dim)

        # Label correlation learning
        self.label_correlation = nn.Parameter(torch.randn(num_labels, num_labels))

        # Final prediction layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

    def forward(self, sample_feature, label_feature):
        batch_size = sample_feature.shape[0]

        # Attention mechanism
        query = self.query_proj(sample_feature)  # [batch_size, 1, hidden_dim]
        key = self.key_proj(label_feature).unsqueeze(0)  # [1, num_labels, hidden_dim]
        value = self.value_proj(label_feature).unsqueeze(0)  # [1, num_labels, hidden_dim]

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.feature_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_probs, value)  # [batch_size, 1, hidden_dim]

        # Combine sample feature with context vector
        combined_feature = torch.cat([sample_feature, context_vector],
                                     dim=-1)  # [batch_size, 1, hidden_dim*2]

        # Final prediction
        hidden = F.relu(self.fc1(combined_feature))
        logits = self.fc2(hidden).squeeze(-1)  # [batch_size, num_labels]

        # Apply label correlation
        corr_logits = torch.matmul(logits, self.label_correlation)
        final_logits = logits + corr_logits

        return final_logits

    
class LatentDecoder(nn.Module):
    def __init__(self, latent_dim, emb_size):
        super(LatentDecoder, self).__init__()

        self.fd = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, emb_size),
            nn.LeakyReLU()
        )

    def forward(self, latent):
        d = self.fd(latent)
        d = F.normalize(d, dim=1)

        return d
    
# class AttentionDecoder(nn.Module):
#     def __init__(self, d_in, d_model, num_labels, hidden_dim=512):
#         super(AttentionDecoder, self).__init__()
#         self.d_in = d_in
#         self.d_model = d_model
#         self.hidden_dim = hidden_dim
#         self.num_labels = num_labels

#         # Attention mechanism
#         self.query_proj = nn.Linear(d_in, hidden_dim)
#         self.key_proj = nn.Linear(d_model, hidden_dim)
#         self.value_proj = nn.Linear(d_model, hidden_dim)

#         self.label_correlation = nn.Parameter(torch.randn(num_labels, num_labels))
#         # Final prediction layers
#         self.fc1 = nn.Linear(hidden_dim + d_in, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, num_labels)

#     def forward(self, samples, x, embs):
#         batch_size = samples.shape[0]
#         samples = torch.cat([samples, x], dim=1).unsqueeze(1)  # [batch_size, 1, d_model]

#         # Attention mechanism
#         query = self.query_proj(samples)  # [batch_size, 1, hidden_dim]
#         key = self.key_proj(embs).unsqueeze(0)  # [1, num_labels, hidden_dim]
#         value = self.value_proj(embs).unsqueeze(0)  # [1, num_labels, hidden_dim]

#         attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
#         attention_probs = F.softmax(attention_scores, dim=-1)
#         context_vector = torch.matmul(attention_probs, value)  # [batch_size, 1, hidden_dim]

#         # Combine sample feature with context vector
#         combined_feature = torch.cat([samples, context_vector],
#                                      dim=-1)  # [batch_size, 1, hidden_dim*2]

#         # Final prediction
#         hidden = F.relu(self.fc1(combined_feature))
#         logits = self.fc2(hidden).squeeze(1)  # [batch_size, num_labels]
#         logits = logits + torch.matmul(logits, self.label_correlation)
#         return logits
    

class AttentionDecoder(nn.Module):
    def __init__(self, d_in, d_model, num_labels):
        super(AttentionDecoder, self).__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.num_labels = num_labels
        self.in_proj = nn.Linear(d_in, d_model)
        # Attention mechanism
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Final prediction layers
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(num_labels)
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, num_labels)

    def forward(self, samples, x, embs):
        samples = torch.cat([samples, x], dim=1).unsqueeze(1)  # [batch_size, 1, d_model]
        q_samples = self.in_proj(samples)  # [batch_size, 1, d_model]

        # Attention mechanism
        query = self.query_proj(q_samples)  # [batch_size, 1, d_model]
        key = self.key_proj(embs).unsqueeze(0)  # [1, num_labels, d_model]
        value = self.value_proj(embs).unsqueeze(0)  # [1, num_labels, d_model]

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attn_output = torch.matmul(attention_probs, value)  # [batch_size, 1, d_model]

        # Combine sample feature with context vector
        attn_output = self.ln1(attn_output + q_samples)

        # Final prediction
        out = F.gelu(self.fc1(attn_output))
        logits = self.fc2(out).squeeze(1)  # [batch_size, num_labels]
        return logits, attention_probs


class GraphDecoder(nn.Module):
    def __init__(self, d_in, n_layers=6, n_head=8, n_head2=8, d_k=64, d_v=64,
                 d_model=512, d_inner_hid=1024, dropout=0.1, dropout2=0.1, 
                 no_dec_self_attn=False, label_adj_matrix=None, num_labels=10, attn_type='softmax'):
        super(GraphDecoder, self).__init__()
        self.d_in = d_in
        self.constant_input = torch.from_numpy(np.arange(num_labels)).view(-1,1)
        # self.in_proj = nn.Linear(d_in, d_model)
        if label_adj_matrix is not None:
            for i in range(label_adj_matrix.size(0)):
                if label_adj_matrix[i].sum().item() < 1:
                    label_adj_matrix[i,i] = 1 #This prevents Nan output in attention (otherwise 0 attn weights occurs)
            self.label_mask = utils.swap_0_1(label_adj_matrix,1,0).unsqueeze(0)
        else:   
            self.label_mask = None
        
        self.layer_stack = nn.ModuleList()
        for _ in range(n_layers):
            self.layer_stack.append(DecoderLayer(d_model, d_inner_hid, n_head,n_head2, d_k, d_v, dropout=dropout,dropout2=dropout2,no_dec_self_att=no_dec_self_attn,attn_type=attn_type))           
        self.tgt_word_proj = XavierLinear(d_model, 1)


    def forward(self, enc_output, src_seq, embs):
        if len(enc_output.size()) == 2:
            enc_output = enc_output.unsqueeze(1)
        batch_size = src_seq.size(0)
        dec_input = embs.repeat(batch_size,1,1) # [batch_size, num_labels, d_model]
        tgt_seq = self.constant_input.repeat(1,batch_size).transpose(0,1).to(src_seq.device) # [batch_size, num_labels]
        dec_enc_attn_pad_mask = utils.get_attn_padding_mask(tgt_seq, src_seq[:,0:enc_output.size(1)])
        if self.label_mask is not None:
            dec_slf_attn_mask = self.label_mask.repeat(batch_size,1,1).to(src_seq.device).byte()
        else:
            dec_slf_attn_mask = None
        dec_output = dec_input
        for idx,dec_layer in enumerate(self.layer_stack):
            dec_output, dec_output_int, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output,slf_attn_mask=dec_slf_attn_mask,dec_enc_attn_mask=dec_enc_attn_pad_mask)
        logits = self.tgt_word_proj(dec_output).squeeze(-1) # [batch_size, num_labels]
        return logits, dec_slf_attn