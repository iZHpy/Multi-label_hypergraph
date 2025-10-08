import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import utils.evals as evals
import utils.utils as utils
from utils.data_loader import process_data
import torch, torch.nn as nn, torch.nn.functional as F
import Hyperlabel.Constants as Constants
from Hyperlabel.Models import Hyperlabel, compute_loss
from Hyperlabel.Translator import translate
from config_args import config_args,get_args
from pdb import set_trace as stop
from tqdm import tqdm


def print_decoder_grad(decoder,nll_loss,nll_loss_x):
    g_x = torch.autograd.grad(nll_loss, decoder.parameters(), retain_graph=True, allow_unused=True)
    g_e = torch.autograd.grad(nll_loss_x, decoder.parameters(), retain_graph=True, allow_unused=True)
    num, den = 0.0, 1e-12
    for gx, ge in zip(g_x, g_e):
        if gx is None or ge is None: continue
        num += (gx.flatten() @ ge.flatten()).item()
        den += (gx.flatten().norm() * ge.flatten().norm()).item()
    cosine = num / den
    print("grad cosine between BCE_x and BCE_e on decoder:", cosine)

def train_epoch(model,train_data, crit, optimizer,scheduler, epoch,opt):
    model.train()

    out_len = opt.tgt_vocab_size

    all_predictions = torch.zeros(len(train_data._src_insts),out_len)
    all_targets = torch.zeros(len(train_data._src_insts),out_len)

    batch_idx,batch_size = 0,train_data._batch_size
    total_loss,total_nll_loss,total_nll_loss_x,total_kl_loss,total_cpc_loss = 0,0,0,0,0

    start_idx, end_idx = (batch_idx * batch_size), ((batch_idx + 1) * batch_size)
    for batch in tqdm(train_data, mininterval=0.5,desc='(Training)', leave=False):
        src,adj,tgt = batch
        loss,d_loss = 0,0
        gold = tgt[:, 1:]

        gold_binary = utils.get_gold_binary(gold.data.cpu(),opt.tgt_vocab_size).to(opt.device)
        optimizer.zero_grad()
        output, attn_x, attn_e = model(src,adj,gold_binary,start_idx, end_idx)
        sum_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, _, logits = \
                    compute_loss(gold_binary, output, opt)
        loss += sum_loss
        total_loss += sum_loss.item()
        total_nll_loss += nll_loss.item()
        total_nll_loss_x += nll_loss_x.item()
        total_kl_loss += kl_loss.item()
        total_cpc_loss += cpc_loss.item()

        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        tgt_out = gold_binary.data
        pred_out = torch.sigmoid(logits).data

        ## Collect batch predictions and targets ##
        all_predictions[start_idx:end_idx] = pred_out
        all_targets[start_idx:end_idx] = tgt_out
        ## Updates ##
        batch_idx +=1
        start_idx, end_idx = (batch_idx*batch_size),((batch_idx+1)*batch_size)
        if end_idx > len(train_data._src_insts):
            end_idx = len(train_data._src_insts)
    # label_features represents the encoded label information after each epoch training
    return all_predictions, all_targets, (total_loss, total_nll_loss, total_nll_loss_x, total_kl_loss, total_cpc_loss)