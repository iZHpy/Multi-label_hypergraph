import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Hyperlabel.Constants as Constants
from pdb import set_trace as stop 


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    pe = torch.zeros(n_position, d_pos_vec)
    position = torch.arange(0, n_position).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_pos_vec, 2) * -(np.log(10000.0) / d_pos_vec))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    pe[0] = 0
    return pe.type(torch.FloatTensor)


def onehot_init(n_src_vocab, d_word_vec):    
    return torch.from_numpy(position_enc).type(torch.FloatTensor)



def get_attn_padding_mask(seq_q, seq_k, unsqueeze=True):
    ''' Indicate the padding-related part to mask '''
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)   # bx1xsk
    if unsqueeze:
        pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k) # bxsqxsk
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    ''' Get an attention mask to avoid using the subsequent info.'''
    assert seq.dim() == 2
    attn_shape = (seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()
    return subsequent_mask

def swap_0_1(tensor, on_zero, on_non_zero):
    res = tensor.clone()
    res[tensor==0] = on_zero
    res[tensor!=0] = on_non_zero
    return res
