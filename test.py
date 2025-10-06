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


def test_epoch(model, test_data,opt, description):
	model.eval()
	out_len = (opt.tgt_vocab_size)
	all_predictions = torch.zeros(len(test_data._src_insts),out_len)
	all_targets = torch.zeros(len(test_data._src_insts),out_len)
	batch_idx = 0
	batch_size = test_data._batch_size
	total_loss, total_nll_loss, total_nll_loss_x, total_kl_loss, total_cpc_loss = 0,0,0,0,0

	start_idx, end_idx = (batch_idx*batch_size), ((batch_idx+1)*batch_size)
	for batch in tqdm(test_data, mininterval=0.5, desc=description, leave=False):
		src,adj,tgt = batch
		gold = tgt[:, 1:]
		
		pad_batch = False
		if opt.multi_gpu and (batch[0][0].size(0) < opt.batch_size):
			pad_batch = True
		if pad_batch:
			diff = opt.batch_size - src[0].size(0)
			src = [torch.cat((src[0],torch.zeros(diff,src[0].size(1)).type(src[0].type()).to(opt.device)),0),
				   torch.cat((src[1],torch.zeros(diff,src[1].size(1)).type(src[1].type()).to(opt.device)),0),
				   torch.cat((src[2],torch.zeros(diff,src[2].size(1)).type(src[2].type()).to(opt.device)),0)]
			tgt = torch.cat((tgt,torch.zeros(diff,tgt.size(1)).type(tgt.type()).to(opt.device)),0)
		gold_binary = utils.get_gold_binary(gold.data.cpu(),opt.tgt_vocab_size).to(opt.device)
		output = model(src,adj,gold_binary,start_idx, end_idx)
		sum_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, _, feat_out = \
                    compute_loss(gold_binary, output, opt)

		if pad_batch:
			feat_out = feat_out[0:batch[0][0].size(0)]
			gold = gold[0:batch[0][0].size(0)]

		total_loss += sum_loss.item()
		total_nll_loss += nll_loss.item()
		total_nll_loss_x += nll_loss_x.item()
		total_kl_loss += kl_loss.item()
		total_cpc_loss += cpc_loss.item()
		feat_out = torch.sigmoid(feat_out).data
		gold_binary = gold_binary.data

		# if start_idx == 0:
		# 	print(gold[0])
		# 	print(gold_binary[0])
		# 	print(feat_out[gold_binary==0])
		# 	print(feat_out[gold_binary==1])
		# 	print('++++++++++++++++++++++++++++++++++')
	
		all_predictions[start_idx:end_idx] = feat_out
		all_targets[start_idx:end_idx] = gold_binary
			
		batch_idx+=1
		start_idx, end_idx = (batch_idx*batch_size),((batch_idx+1)*batch_size)
		if end_idx > len(test_data._src_insts):
			end_idx = len(test_data._src_insts)
  
	
	return all_predictions, all_targets, (total_loss, total_nll_loss, total_nll_loss_x, total_kl_loss, total_cpc_loss)

