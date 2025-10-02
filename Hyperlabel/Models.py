import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Hyperlabel.Constants as Constants
from Hyperlabel.Layers import EncoderLayer,DecoderLayer
from Hyperlabel.SubLayers import ScaledDotProductAttention
from Hyperlabel.SubLayers import PositionwiseFeedForward
from Hyperlabel.SubLayers import XavierLinear
from Hyperlabel.Encoders import MLPEncoder,GraphEncoder,RNNEncoder
from Hyperlabel.Decoders import MLPDecoder,RNNDecoder,GraphDecoder
from Hyperlabel.Loss import AsymmetricLoss, FocalLoss, kl_align_samples_as_gauss, kl_latents_norm, kl_latents_as_logits, js_divergence
from pdb import set_trace as stop 
from Hyperlabel import utils
import copy
from hyper.hypergraph import WeightedHypergraph
from hyper.models import WeightedHypergraphModel
from hyper.decoder import SimpleDecoder, ComplexDecoder, LatentDecoder, AttentionDecoder


class Hyperlabel(nn.Module):
    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_seq_e, train_labels, n_layers_sample_enc=6,
            n_layers_label_enc=6,n_head=8,d_word_vec=512, d_model=512, d_emb=512, d_inner_hid=1024, d_latent=64,
            d_k=64, d_v=64,sample_enc_dropout=0.1, label_enc_dropout=0.1,decoder_type='simple',enc_transform='special_token',
            special_token_init='default', feature_aggregate='mean', node2hyperedge_aggregate='mean', node_update='simple',feat_mode='tokens',no_enc_pos_embedding=False):

        super(Hyperlabel, self).__init__()
        self.feat_mode = feat_mode
        self.n_src_vocab = n_src_vocab

        self.hypergraph = WeightedHypergraph(num_labels=n_tgt_vocab)
        self.hypergraph.create_from_labels(labels=train_labels)
        
        ############# Sample Encoder ###########
        if feat_mode == 'float':
            self.sample_encoder = MLPEncoder( 
                n_src_vocab, n_max_seq_e, n_layers=n_layers_sample_enc, n_head=n_head,
                d_word_vec=d_word_vec, d_model=d_model, d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid, feat_mode='float', dropout=sample_enc_dropout)
        else:
            self.sample_encoder = GraphEncoder( 
                n_src_vocab, n_max_seq_e, n_layers=n_layers_sample_enc, n_head=n_head,
                d_word_vec=d_word_vec, d_model=d_model, d_latent=d_latent, d_k=d_k, d_v=d_v,
                d_inner_hid=d_inner_hid, feat_mode=feat_mode, dropout=sample_enc_dropout,
                no_enc_pos_embedding=no_enc_pos_embedding,enc_transform=enc_transform,
                special_token_init=special_token_init)

        ############# Label Encoder ###########
        self.label_embedding = nn.Embedding(n_tgt_vocab, d_model)
        self.dropout = nn.Dropout(label_enc_dropout)
        
        self.label_encoder = WeightedHypergraphModel(num_labels=n_tgt_vocab, feature_dim=d_model, d_latent=d_latent, dropout_rate=label_enc_dropout ,
            num_layers=n_layers_label_enc, feature_aggregate=feature_aggregate, node2hyperedge_aggregate=node2hyperedge_aggregate, node_update=node_update,num_heads=n_head)
        
        ############# Decoder ###########
        self.decoder = AttentionDecoder(d_latent=d_latent, num_labels=n_tgt_vocab, hidden_dim=d_model)

        # self.bias_x = nn.Parameter(self.generate_bias(train_labels, n_tgt_vocab))  # learnable bias
        # self.bias_e = nn.Parameter(self.generate_bias(train_labels, n_tgt_vocab))
        # self.log_tau = nn.Parameter(torch.log(torch.tensor(0.1)))
        
    def generate_bias(self, train_labels, n_tgt_vocab, n_samples=None):
        # train_labels: list of lists (positives per sample)
        label_pos = np.zeros(n_tgt_vocab, dtype=np.int64)
        for labels in train_labels:
            for l in labels:
                label_pos[l] += 1
        if n_samples is None:
            n_samples = len(train_labels)
        pi = label_pos / max(n_samples, 1)   # 每类正例先验
        pi = np.clip(pi, 1e-4, 1 - 1e-4)
        bias = torch.log(torch.tensor(pi) / (1 - torch.tensor(pi)))
        return bias
    
    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        freezed_param_ids = set()
        if hasattr(self.sample_encoder, 'position_enc'):
            enc_freezed_param_ids = set(map(id, self.sample_encoder.position_enc.parameters()))
            freezed_param_ids = freezed_param_ids | enc_freezed_param_ids
        if self.feat_mode == 'onehot':
            enc_onehot_param_ids = set(map(id, self.sample_encoder.src_word_emb.parameters()))
            freezed_param_ids = freezed_param_ids | enc_onehot_param_ids
    
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def feat_forward(self, src_seq, adj, src_pos):
        feat_latent = self.sample_encoder(src_seq, adj, src_pos).squeeze(1)
        feat_out = {'feat_latent': feat_latent}
        return feat_out

    def label_forward(self, binary_tgt, feat_latent, start_index, end_index):
        h0 = self.dropout(F.relu(self.label_embedding.weight))  # (num_labels, d_model)
        label_space, _ = self.label_encoder(hypergraph=self.hypergraph, batch_features=feat_latent, node_features=h0, start_index=start_index, end_index=end_index, device=feat_latent.device)
        label_latent = torch.matmul(binary_tgt, label_space) / binary_tgt.sum(1, keepdim=True)
        label_out = {'label_latent': label_latent, 'label_space': label_space}
        return label_out

    def forward(self, src, adj, binary_tgt, start_index, end_index):
        src_seq, src_pos = src

        # sample_encode
        fx_out = self.feat_forward(src_seq, adj, src_pos)
        feat_latent = fx_out['feat_latent']
         
        # label_encode
        fe_out = self.label_forward(binary_tgt, feat_latent, start_index, end_index)
        label_latent = fe_out['label_latent']
        # decode
        label_space = fe_out['label_space']
        embs = self.label_embedding.weight
        
        logits_x = self.decoder(feat_latent, embs)
        logits_e = self.decoder(label_latent, embs)
        # label_out = cosine_logits(label_emb, embs, log_tau=self.log_tau, bias=self.bias_e.to(label_emb.device))
        # feat_out = cosine_logits(feat_emb, embs, log_tau=self.log_tau, bias=self.bias_x.to(feat_emb.device))
        # label_out = torch.matmul(label_emb, embs)
        # feat_out = torch.matmul(feat_emb, embs)
        
        output = fe_out
        output.update(fx_out)
        output['embs'] = embs
        output['logits_e'] = logits_e
        output['logits_x'] = logits_x

        return output

def cosine_logits(x_emb, y_emb, log_tau=0, bias=None):
    x_norm = F.normalize(x_emb, p=2, dim=-1)
    y_norm = F.normalize(y_emb, p=2, dim=0)
    scale = 1.0 / (torch.exp(log_tau) if log_tau is not None else 1.0)
    logits = torch.matmul(x_norm, y_norm) * scale
    if bias is not None:
        logits = logits + bias
    return logits
    
def compute_loss(input_label, output, args=None):
    logits_e, label_space,  label_latent = \
        output['logits_e'], output['label_space'],  output['label_latent']
    logits_x, feat_latent = \
        output['logits_x'], output['feat_latent']

    # kl_loss = utils.kl_align_samples_as_gauss(label_latent, feat_latent, tau=1.0, reduction='mean')
    
    kl_loss = kl_latents_as_logits(label_latent, feat_latent, tau=1.0)

    def supconloss(logits_e, logits_x):
    
        labels = torch.cat((input_label, input_label)).float()
        n_label = labels.shape[1]
        emb_labels = torch.eye(n_label).to(labels.device)
        mask = torch.matmul(labels, emb_labels)

        anchor_dot_contrast = torch.cat([logits_e, logits_x], dim=0)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

    nll_loss = nn.BCEWithLogitsLoss()(logits_e, input_label)
    nll_loss_x = nn.BCEWithLogitsLoss()(logits_x, input_label)
    sum_nll_loss = nll_loss + nll_loss_x
    cpc_loss = supconloss(logits_e, logits_x)
    sum_loss = sum_nll_loss + kl_loss + cpc_loss
    return sum_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, logits_e, logits_x

