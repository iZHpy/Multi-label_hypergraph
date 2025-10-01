import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lamp.Constants as Constants
from lamp.Layers import EncoderLayer,DecoderLayer
from lamp.SubLayers import ScaledDotProductAttention
from lamp.SubLayers import PositionwiseFeedForward
from lamp.SubLayers import XavierLinear
from lamp.Encoders import MLPEncoder,GraphEncoder,RNNEncoder
from lamp.Decoders import MLPDecoder,RNNDecoder,GraphDecoder
from pdb import set_trace as stop 
from lamp import utils
import copy
from hyper.hypergraph import WeightedHypergraph
from hyper.models import WeightedHypergraphModel
from hyper.decoder import SimpleDecoder, ComplexDecoder, LatentDecoder
 

class LAMP(nn.Module):
    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_seq_e, train_labels, n_layers_sample_enc=6,
            n_layers_label_enc=6,n_head=8,d_word_vec=512, d_model=512, d_emb=512, d_inner_hid=1024, d_latent=64,
            d_k=64, d_v=64,sample_enc_dropout=0.1, label_enc_dropout=0.1,decoder_type='simple',enc_transform='special_token',
            special_token_init='default', feature_aggregate='mean', node2hyperedge_aggregate='mean', node_update='simple',onehot=False,no_enc_pos_embedding=False):

        super(LAMP, self).__init__()
        self.onehot = onehot
        self.n_src_vocab = n_src_vocab

        self.hypergraph = WeightedHypergraph(num_labels=n_tgt_vocab)
        self.hypergraph.create_from_labels(labels=train_labels)
        
        ############# Sample Encoder ###########
        self.sample_encoder = GraphEncoder( 
            n_src_vocab, n_max_seq_e, n_layers=n_layers_sample_enc, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model, d_latent=d_latent, d_k=d_k, d_v=d_v,
            d_inner_hid=d_inner_hid, onehot=onehot, dropout=sample_enc_dropout,
            no_enc_pos_embedding=no_enc_pos_embedding,enc_transform=enc_transform,
            special_token_init=special_token_init)

        ############# Label Encoder ###########
        self.label_embedding = nn.Linear(n_tgt_vocab, d_model)
        self.dropout = nn.Dropout(label_enc_dropout)
        
        self.label_encoder = WeightedHypergraphModel(num_labels=n_tgt_vocab, feature_dim=d_model, d_latent=d_latent, dropout_rate=label_enc_dropout ,
            num_layers=n_layers_label_enc, feature_aggregate=feature_aggregate, node2hyperedge_aggregate=node2hyperedge_aggregate, node_update=node_update,num_heads=n_head)
        
        ############# Decoder ###########
        self.decoder = LatentDecoder(latent_dim=d_latent, emb_size=d_emb)
        # if decoder_type=='simple':
        #     self.decoder = SimpleDecoder(feature_dim=d_model, num_labels=n_tgt_vocab)
        # elif decoder_type=='complex':
        #     self.decoder = ComplexDecoder(feature_dim=d_model, num_labels=n_tgt_vocab)
        # else:
        #     raise NotImplementedError


    def get_trainable_parameters(self):
        ''' Avoid updating the position encoding '''
        freezed_param_ids = set()
        if hasattr(self.sample_encoder, 'position_enc'):
            enc_freezed_param_ids = set(map(id, self.sample_encoder.position_enc.parameters()))
            freezed_param_ids = freezed_param_ids | enc_freezed_param_ids
        if self.onehot:
            enc_onehot_param_ids = set(map(id, self.sample_encoder.src_word_emb.parameters()))
            freezed_param_ids = freezed_param_ids | enc_onehot_param_ids
    
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def feat_forward(self, src_seq, adj, src_pos, x):
        feat_latent = self.sample_encoder(src_seq, adj, src_pos).squeeze(1)
        feat_emb = self.decoder(feat_latent)
        feat_out = {'feat_latent': feat_latent, 'feat_emb': feat_emb}
        return feat_out

    def label_forward(self, x, binary_tgt, feat_latent, start_index, end_index):
        n_label = self.hypergraph.num_labels
        all_labels = torch.eye(n_label).to(feat_latent.device)
        h0 = self.dropout(F.relu(self.label_embedding(all_labels)))
        label_space, _ = self.label_encoder(hypergraph=self.hypergraph, batch_features=feat_latent, node_features=h0, start_index=start_index, end_index=end_index, device=feat_latent.device)
        label_latent = torch.matmul(binary_tgt, label_space) / binary_tgt.sum(1, keepdim=True)
        label_emb = self.decoder(label_latent)
        label_out = {'label_latent': label_latent, 'label_emb': label_emb, 'label_space': label_space}
        return label_out

    def forward(self, src, adj, binary_tgt, start_index, end_index):
        src_seq, src_pos, src_onehot = src

        # sample_encode
        fx_out = self.feat_forward(src_seq, adj, src_pos, src_onehot)
        feat_latent = fx_out['feat_latent']
        feat_emb = fx_out['feat_emb']
         
        # label_encode
        fe_out = self.label_forward(src_onehot, binary_tgt, feat_latent, start_index, end_index)
        label_emb = fe_out['label_emb']
        # decode
        embs = self.label_embedding.weight
        label_out = torch.matmul(label_emb, embs)
        feat_out = torch.matmul(feat_emb, embs)
        
        output = fe_out
        output.update(fx_out)
        output['embs'] = embs
        output['label_out'] = label_out
        output['feat_out'] = feat_out

        return output
    
    
def compute_loss(input_label, output, args=None):
    label_out, label_space, label_emb, label_latent = \
        output['label_out'], output['label_space'], output['label_emb'], output['label_latent']
    feat_out, feat_emb, feat_latent = \
        output['feat_out'], output['feat_emb'], output['feat_latent']
    embs = output['embs']

    # kl_loss = utils.kl_align_samples_as_gauss(label_latent, feat_latent, tau=1.0, reduction='mean')
    
    kl_loss = utils.kl_latents_as_logits(label_latent, feat_latent, tau=1.0)

    def supconloss(label_emb, feat_emb, embs, temp=1.0):
        features = torch.cat((label_emb, feat_emb))
        labels = torch.cat((input_label, input_label)).float()
        n_label = labels.shape[1]
        emb_labels = torch.eye(n_label).to(labels.device)
        mask = torch.matmul(labels, emb_labels)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, embs),
            temp)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

    nll_loss = F.binary_cross_entropy_with_logits(label_out, input_label)
    nll_loss_x = F.binary_cross_entropy_with_logits(feat_out, input_label)
    sum_nll_loss = nll_loss + nll_loss_x
    cpc_loss = supconloss(label_emb, feat_emb, embs)
    sum_loss = sum_nll_loss * args.nll_coeff + kl_loss * 1. + cpc_loss
    return sum_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, label_out, feat_out

