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
from hyper.decoder import SimpleDecoder, ComplexDecoder
 

class LAMP(nn.Module):
    def __init__(
            self, n_src_vocab, n_tgt_vocab, n_max_seq_e, train_labels, n_layers_sample_enc=6,
            n_layers_label_enc=6,n_head=8,d_word_vec=512, d_model=512, d_inner_hid=1024,
            d_k=64, d_v=64,sample_enc_dropout=0.1, label_enc_dropout=0.1,decoder_type='simple',enc_transform='special_token',
            special_token_init='default', feature_aggregate='mean', node2hyperedge_aggregate='mean', node_update='simple',onehot=False,no_enc_pos_embedding=False):

        super(LAMP, self).__init__()
        self.onehot = onehot

        self.hypergraph = WeightedHypergraph(num_labels=n_tgt_vocab)
        self.hypergraph.create_from_labels(labels=train_labels)
        
        ############# Sample Encoder ###########
        self.sample_encoder = GraphEncoder( 
            n_src_vocab, n_max_seq_e, n_layers=n_layers_sample_enc, n_head=n_head,
            d_word_vec=d_word_vec, d_model=d_model,d_k=d_k, d_v=d_v,
            d_inner_hid=d_inner_hid, onehot=onehot, dropout=sample_enc_dropout,
            no_enc_pos_embedding=no_enc_pos_embedding,enc_transform=enc_transform,
            special_token_init=special_token_init)

        ############# Label Encoder ###########

        self.label_encoder = WeightedHypergraphModel(num_labels=n_tgt_vocab, feature_dim=d_model, dropout_rate=label_enc_dropout ,
            num_layers=n_layers_label_enc, feature_aggregate=feature_aggregate, node2hyperedge_aggregate=node2hyperedge_aggregate, node_update=node_update,num_heads=n_head)
        
        ############# Decoder ###########

        if decoder_type=='simple':
            self.decoder = SimpleDecoder(feature_dim=d_model, num_labels=n_tgt_vocab)
        elif decoder_type=='complex':
            self.decoder = ComplexDecoder(feature_dim=d_model, num_labels=n_tgt_vocab)
        else:
            raise NotImplementedError


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

    def initialize_cache(self, num_samples):
        self.cache_samples = torch.zeros(num_samples, self.sample_encoder.d_model)

    def cache_samples_func(self, src, adj, start_idx, end_idx):
        src_seq, src_pos = src
        sample_features = self.sample_encoder(src_seq, adj, src_pos)
        if self.training:
            self.cache_samples[start_idx:end_idx] = sample_features.squeeze(1).detach() 
        return sample_features
        
    def forward(self, src, adj, label_features, binary_tgt,start_index, end_index):
        src_seq, src_pos = src

        sample_features = self.sample_encoder(src_seq, adj, src_pos)
        # sample_features = self.cache_samples_func(src, adj, start_index, end_index)
        if label_features is None:
            label_features, _ = self.label_encoder(self.hypergraph, sample_features, start_index, end_index, device=sample_features.device)

        # raise NotImplementedError
        logits = self.decoder(sample_features, label_features).squeeze(1)

        return logits, label_features
