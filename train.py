import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import utils.evals as evals
import utils.utils as utils
from utils.data_loader import process_data
import torch, torch.nn as nn, torch.nn.functional as F
import lamp.Constants as Constants
from lamp.Models import LAMP, compute_loss
from lamp.Translator import translate
from config_args import config_args,get_args
from pdb import set_trace as stop
from tqdm import tqdm



def train_epoch(model,train_data, crit, optimizer,adv_optimizer,epoch,data_dict,opt):
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
		output = model(src,adj,gold_binary,start_idx, end_idx)
		sum_loss, nll_loss, nll_loss_x, kl_loss, cpc_loss, _, feat_out = \
                    compute_loss(gold_binary, output, opt)
		loss += sum_loss
		total_loss += sum_loss.item()
		total_nll_loss += nll_loss.item()
		total_nll_loss_x += nll_loss_x.item()
		total_kl_loss += kl_loss.item()
		total_cpc_loss += cpc_loss.item()
		loss.backward()
		optimizer.step()
		tgt_out = gold_binary.data
		pred_out = torch.sigmoid(feat_out).data

		## Updates ##
		# start_idx, end_idx = (batch_idx*batch_size),((batch_idx+1)*batch_size)
		all_predictions[start_idx:end_idx] = pred_out
		all_targets[start_idx:end_idx] = tgt_out
		batch_idx +=1
		start_idx, end_idx = (batch_idx*batch_size),((batch_idx+1)*batch_size)
		if end_idx > len(train_data._src_insts):
			end_idx = len(train_data._src_insts)
		
	# label_features represents the encoded label information after each epoch training
	return all_predictions, all_targets, (total_loss, total_nll_loss, total_nll_loss_x, total_kl_loss, total_cpc_loss)