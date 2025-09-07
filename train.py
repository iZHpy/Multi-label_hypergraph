import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import utils.evals as evals
import utils.utils as utils
from utils.data_loader import process_data
import torch, torch.nn as nn, torch.nn.functional as F
import lamp.Constants as Constants
from lamp.Models import LAMP
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
	bce_total,d_total,d_fake_total,g_total = 0,0,0,0

	start_idx, end_idx = (batch_idx * batch_size), ((batch_idx + 1) * batch_size)
	for batch in tqdm(train_data, mininterval=0.5,desc='(Training)', leave=False):
		src,adj,tgt = batch
		loss,d_loss = 0,0
		gold = tgt[:, 1:]

		gold_binary = utils.get_gold_binary(gold.data.cpu(),opt.tgt_vocab_size).to(opt.device)
		optimizer.zero_grad()
		pred, label_features = model(src,adj,None,gold_binary,start_idx, end_idx)
		norm_pred = F.sigmoid(pred)
		bce_loss =  F.binary_cross_entropy_with_logits(pred, gold_binary,reduction='mean')
		loss += bce_loss
		bce_total += bce_loss.item()
		loss.backward()
		optimizer.step()
		tgt_out = gold_binary.data
		pred_out = norm_pred.data

		
		## Updates ##
		start_idx, end_idx = (batch_idx*batch_size),((batch_idx+1)*batch_size)
		all_predictions[start_idx:end_idx] = pred_out
		all_targets[start_idx:end_idx] = tgt_out
		batch_idx +=1
		
	# label_features represents the encoded label information after each epoch training
	return all_predictions, all_targets, bce_total, label_features