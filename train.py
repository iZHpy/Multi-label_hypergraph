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


def layer_grad_norms(model):
    report = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            report[name] = {
                "grad_norm": p.grad.data.norm(p=2).item(),
                "abs_mean": p.grad.data.abs().mean().item(),
                "max": p.grad.data.abs().max().item()
            }
    return report

@torch.no_grad()
def _safe_mean(x): 
    return x.mean().item() if x.numel() > 0 else 0.0

def logits_grad_stats(logits, targets, pos_weight=None, neg_easy_q=0.2, neg_hard_q=0.8):
    logits = logits.detach().requires_grad_(True)
    targets = targets.to(dtype=logits.dtype, device=logits.device)

    loss = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="mean"
    )
    (g,) = torch.autograd.grad(loss, logits, retain_graph=False, create_graph=False)

    pos_mask = targets == 1
    neg_mask = targets == 0

    # 用分位数划分负样本难度
    neg_logits = logits.detach()[neg_mask]
    if neg_logits.numel() > 0:
        q_easy = torch.quantile(neg_logits, neg_easy_q)
        q_hard = torch.quantile(neg_logits, neg_hard_q)
        easy_neg_mask = neg_mask & (logits.detach() <= q_easy)
        hard_neg_mask = neg_mask & (logits.detach() >= q_hard)
        mid_neg_mask  = neg_mask & ~(easy_neg_mask | hard_neg_mask)
    else:
        easy_neg_mask = hard_neg_mask = mid_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)

    def sign_rate(mask, expect_positive: bool):
        if not mask.any():
            return 0.0
        gi = g[mask]
        return (gi > 0).float().mean().item() if expect_positive else (gi < 0).float().mean().item()

    out = dict(
        prob_mean=torch.sigmoid(logits.detach()).mean().item(),
        g_mean=g.abs().mean().item(),
        n_total=int(g.numel()),
        n_pos=int(pos_mask.sum().item()),
        n_neg=int(neg_mask.sum().item()),
        n_neg_easy=int(easy_neg_mask.sum().item()),
        n_neg_mid=int(mid_neg_mask.sum().item()),
        n_neg_hard=int(hard_neg_mask.sum().item()),
        g_pos=_safe_mean(g[pos_mask].abs()),
        g_neg_easy=_safe_mean(g[easy_neg_mask].abs()),
        g_neg_mid=_safe_mean(g[mid_neg_mask].abs()),
        g_neg_hard=_safe_mean(g[hard_neg_mask].abs()),
        pos_grad_negative_rate=sign_rate(pos_mask, expect_positive=False),  # y=1 期望负
        neg_grad_positive_rate=sign_rate(neg_mask, expect_positive=True),   # y=0 期望正
        all_negative_grad_rate=(g < 0).float().mean().item(),
    )
    return out

def print_grad(model, logits=None, output=None, gold_binary=None):
	rep = layer_grad_norms(model)
	print('label_emb grad', rep.get("label_embedding.weight", {}))
	print('label_emb b grad', rep.get("label_embedding.bias", {}))
	print('W grad', rep.get("decoder.fd.2.weight", {}))
	print('b grad', rep.get("decoder.fd.2.bias", {}))
	print('Logits grad', logits_grad_stats(output['label_out'], gold_binary))
	print('logits_0', logits[gold_binary==0])	
	print('logits_1', logits[gold_binary==1])
	print('++++++++++++++++++++++++++++++++++')
 
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
        # if (epoch >= 0):
        #     src_seq, src_pos, src_multi_hot = src
        #     for id in range(src_seq.size(0)):
        #         num_true = gold_binary[id].sum().item()
        #         if num_true >= 8:
        #             savedict = { "tgt_raw": gold_binary[id], 
        #                         "attn_e": attn_e[id], "attn_x": attn_x[id].cpu()}
        #             torch.save(savedict, path.join('./label_attn', "epoch{}_idx{}.pt".format(epoch, start_idx+id)))     
        # if (epoch >= 0):
        #     src_seq, src_pos, src_multi_hot = src
        #     for id in range(src_seq.size(0)):
        #         num_true = gold_binary[id].sum().item()
        #         if num_true >= 8:
        #             raw = src_seq[id][src_seq[id]!=0].cpu().numpy().tolist()
        #             l = len(raw)
        #             savedict = {"src_raw": raw, "tgt_raw": gold_binary[id], 
        #                         "attn_e": attn_e[id,:, :l], "attn_x": attn_x[id,:, :l].cpu()}
        #             torch.save(savedict, path.join('./attn_map', "epoch{}_idx{}.pt".format(epoch, start_idx+id)))       
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