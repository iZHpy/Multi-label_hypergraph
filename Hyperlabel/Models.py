import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import Hyperlabel.Constants as Constants
from Hyperlabel.Layers import EncoderLayer,DecoderLayer
from Hyperlabel.SubLayers import ScaledDotProductAttention
from Hyperlabel.SubLayers import PositionwiseFeedForward
from Hyperlabel.SubLayers import XavierLinear
from Hyperlabel.Encoders import MLPEncoder,GraphEncoder,DeepSetsEncoder
from Hyperlabel.SetTransformer import SetTransformerEncoder
from Hyperlabel.Loss import AsymmetricLoss, FocalLoss, kl_align_samples_as_gauss, kl_latents_norm, kl_latents_as_logits, js_divergence
from pdb import set_trace as stop 
from Hyperlabel import utils
import copy
from hyper.hypergraph import WeightedHypergraph
from hyper.models import WeightedHypergraphModel
from hyper.decoder import  LatentDecoder, AttentionDecoder, GraphDecoder



class Hyperlabel(nn.Module):
    def __init__(self, n_src_vocab, n_tgt_vocab, n_max_seq_e, train_labels, n_layers_sample_enc=6,
            n_layers_label_enc=6,n_head=8,n_head2=8,d_word_vec=512, d_model=512, d_inner_hid=1024,
            d_k=64, d_v=64,sample_enc_dropout=0.1, label_enc_dropout=0.1, decoder_type='GraphDecoder',enc_transform='mean',
            special_token_init='default', feature_aggregate='mean', node2hyperedge_aggregate='mean', node_update='simple',feat_mode='tokens',no_enc_pos_embedding=False,
            dec_dropout=0.1,dec_dropout2=0.1,n_layers_dec=2, label_adj_matrix=None):

        super(Hyperlabel, self).__init__()
        self.feat_mode = feat_mode
        self.hypergraph = WeightedHypergraph(num_labels=n_tgt_vocab)
        self.hypergraph.create_from_labels(labels=train_labels)
        self.feat_mode = feat_mode
        self.decoder_type = decoder_type
        
        ############# Sample Encoder ###########
        if feat_mode == 'vec':
            self.sample_encoder = MLPEncoder(d_in=n_src_vocab, d_model=d_model, d_hidden=d_inner_hid,
                                             n_layers=n_layers_sample_enc, dropout=sample_enc_dropout)
        elif feat_mode == 'tokens':
            self.sample_encoder = GraphEncoder(n_src_vocab, n_max_seq_e, n_layers=n_layers_sample_enc, n_head=n_head,
                                               d_word_vec=d_word_vec, d_model=d_model, d_k=d_k, d_v=d_v,
                                               d_inner_hid=d_inner_hid, feat_mode=feat_mode, dropout=sample_enc_dropout,
                                               no_enc_pos_embedding=no_enc_pos_embedding,enc_transform=enc_transform,
                                               special_token_init=special_token_init)
        else:
            self.sample_encoder = MLPEncoder(d_in=n_src_vocab, d_model=d_model, d_hidden=d_inner_hid,
                                                 n_layers=n_layers_sample_enc, dropout=sample_enc_dropout)
            
        ############# Label Encoder ###########
        self.label_embedding = nn.Embedding(n_tgt_vocab, d_model)
        self.dropout = nn.Dropout(label_enc_dropout)
        
        self.label_encoder = WeightedHypergraphModel(num_labels=n_tgt_vocab, feature_dim=d_model, dropout_rate=label_enc_dropout ,
            num_layers=n_layers_label_enc, feature_aggregate=feature_aggregate, node2hyperedge_aggregate=node2hyperedge_aggregate, node_update=node_update,num_heads=n_head)
        
        ############# Decoder ###########
        if decoder_type == 'Graph':
            self.decoder = GraphDecoder(d_in=d_model, n_layers=n_layers_dec, n_head=n_head, n_head2=n_head2, d_k=d_k, d_v=d_v,
                 d_model=d_model, d_inner_hid=d_inner_hid, dropout=dec_dropout, dropout2=dec_dropout2,
                 no_dec_self_attn=False, label_adj_matrix=label_adj_matrix, num_labels=n_tgt_vocab)
        else:
            if self.feat_mode == 'tokens':
                n_src_vocab = n_src_vocab - 4
            self.decoder = AttentionDecoder(d_in=d_model+n_src_vocab, d_model=d_model, num_labels=n_tgt_vocab)

    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        freezed_param_ids = set()
        if hasattr(self.sample_encoder, 'position_enc'):
            enc_freezed_param_ids = set(map(id, self.sample_encoder.position_enc.parameters()))
            freezed_param_ids = freezed_param_ids | enc_freezed_param_ids
    
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def feat_forward(self, src_seq, adj, src_pos):
        if self.feat_mode == 'tokens':
            enc_output, feat_latent = self.sample_encoder(src_seq, adj, src_pos)
        else:
            enc_output, feat_latent = self.sample_encoder(src_seq)
        feat_out = {'feat_latent': feat_latent, 'enc_output': enc_output}
        return feat_out

    def label_forward(self, binary_tgt, feat_latent, start_index, end_index):
        # h0 = self.dropout(F.relu(self.label_embedding.weight))  # (num_labels, d_model)
        h0 = self.dropout(self.label_embedding.weight)  # (num_labels, d_model)
        batch_features  = feat_latent if self.training else None
        label_space, _ = self.label_encoder(hypergraph=self.hypergraph, batch_features=batch_features, node_features=h0, start_index=start_index, end_index=end_index, device=feat_latent.device)
        counts = binary_tgt.sum(dim=1, keepdim=True)  # [B, 1]
        has_label = counts > 0
        counts = torch.clamp(counts, min=1.0)                  
        label_latent = torch.matmul(binary_tgt, label_space) / counts * has_label.float()  # [B, d_model]
        label_out = {'label_latent': label_latent, 'label_space': label_space, 'mask': has_label}
        return label_out

    def forward(self, src, adj, binary_tgt, start_index, end_index):
        src_seq, src_pos, src_multi_hot = src
        # sample_encode
        if (self.feat_mode == 'tokens') or (self.feat_mode == 'vec'):
            fx_out = self.feat_forward(src_seq, adj, src_pos)
        else:
            fx_out = self.feat_forward(src_multi_hot, adj, src_pos)

        enc_output = fx_out['enc_output']
        feat_latent = fx_out['feat_latent']

        # label_encode
        fe_out = self.label_forward(binary_tgt, feat_latent, start_index, end_index)
        label_latent = fe_out['label_latent']

        # decode
        label_space = fe_out['label_space']
        

        if self.feat_mode == 'tokens':
            if self.decoder_type == 'Graph':
                logits_x, _ = self.decoder(enc_output, src_seq, label_space)
                logits_e, _ = self.decoder(label_latent, src_seq, label_space)
            else:
                logits_x = self.decoder(feat_latent, src_multi_hot.float(), label_space)
                logits_e = self.decoder(label_latent, src_multi_hot.float(), label_space)
        elif self.feat_mode == 'vec':
            logits_x, _ = self.decoder(feat_latent, src_seq, label_space)
            logits_e, _ = self.decoder(label_latent, src_seq, label_space)
        output = fe_out
        output.update(fx_out)
        output['logits_e'] = logits_e
        output['logits_x'] = logits_x

        return output

def compute_loss(input_label, output, args=None):
    logits_e, label_space,  label_latent, mask = \
        output['logits_e'], output['label_space'],  output['label_latent'], output['mask']
    logits_x, feat_latent = \
        output['logits_x'], output['feat_latent']

    kl_loss = 1 - F.cosine_similarity(
        F.normalize(feat_latent, dim=-1), 
        F.normalize(label_latent, dim=-1), dim=-1
    ).mean()
    
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



    nll_loss = F.binary_cross_entropy_with_logits(logits_e, input_label, reduction='mean')
    nll_loss_x = F.binary_cross_entropy_with_logits(logits_x, input_label, reduction='mean')
    cpc_loss = supconloss(logits_e, logits_x)
    sum_loss = nll_loss * args.nll_e_weight  + nll_loss_x * args.nll_x_weight + cpc_loss * args.cpc_weight # + kl_loss * args.kl_weight
    return sum_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, logits_e, logits_x

