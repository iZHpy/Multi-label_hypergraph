import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import utils.evals as evals
import utils.utils as utils
from utils.data_loader import process_data
import torch, torch.nn as nn, torch.nn.functional as F
import Hyperlabel.Constants as Constants
from Hyperlabel.Models import Hyperlabel
from Hyperlabel.Translator import translate
from Hyperlabel.Constants import THRESHOLDS
from config_args import config_args,get_args
from pdb import set_trace as stop
from tqdm import tqdm
from train import train_epoch
from test import test_epoch
import optuna

warnings.filterwarnings("ignore")

def run_model(model, train_data, valid_data, test_data, crit, optimizer,adv_optimizer,scheduler, opt, trial=None):
	logger = evals.Logger(opt)

	valid_losses = []

	losses = []

	max_metrics = {'ACC':1000000,'HA':0,'ebF1':0,'miF1':0,'maF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0, 'allAUC':None,'allAUPR':None}
	if opt.test_only:
		start = time.time()
		all_predictions, all_targets, test_loss = test_epoch(model, test_data,opt,'(Testing)')
		elapsed = ((time.time()-start)/60)
		print('\n(Testing) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
		test_loss = test_loss/len(test_data._src_insts)
		print('B : '+str(test_loss))

		test_metrics = evals.compute_metrics(all_predictions,all_targets,0,opt,elapsed,all_metrics=True)

		return

	loss_file = open(path.join(opt.model_name,'losses.csv'),'w+')
 
	best_valid_metrics = {'loss':1000000,'ACC':0,'HA':0,'ebF1':0,'miF1':0,'maF1':0,'meanAUC':0,'medianAUC':0,'meanAUPR':0,'medianAUPR':0,'meanFDR':0,'medianFDR':0,'allAUC':None,'allAUPR':None}

	for epoch_i in range(opt.epoch):
		print('================= Epoch', epoch_i+1, '=================')
		################################## TRAIN ###################################
		start = time.time()
		all_predictions,all_targets,train_loss=train_epoch(model,train_data,crit,optimizer,scheduler,(epoch_i+1),opt)
		elapsed = ((time.time()-start)/60)
		print('\n(Training) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
		train_total_loss, train_nll_loss, train_nll_loss_x, train_kl_loss, train_cpc_loss = train_loss
		print('Total_Loss : '+str(train_total_loss/len(train_data._src_insts)))
		print('NLL_Loss : '+str(train_nll_loss/len(train_data._src_insts)))
		print('NLL_X_Loss : '+str(train_nll_loss_x/len(train_data._src_insts)))
		print('KL_Loss : '+str(train_kl_loss/len(train_data._src_insts)))
		print('CPC_Loss : '+str(train_cpc_loss/len(train_data._src_insts)))

		
		# torch.save(all_predictions,path.join(opt.model_name,'epochs','train_preds'+str(epoch_i+1)+'.pt'))
		# torch.save(all_targets,path.join(opt.model_name,'epochs','train_targets'+str(epoch_i+1)+'.pt'))
		train_metrics = evals.compute_metrics(all_predictions,all_targets,0,opt,elapsed,all_metrics=True)  
		################################### VALID ###################################

		start = time.time()
		all_predictions, all_targets,valid_loss = test_epoch(model, valid_data,opt,'(Validation)')
		elapsed = ((time.time()-start)/60)
		print('\n(Validation) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
		valid_total_loss, valid_nll_loss, valid_nll_loss_x, valid_kl_loss, valid_cpc_loss = valid_loss
		print('Total_Loss : '+str(valid_total_loss/len(valid_data._src_insts)))
		print('NLL_Loss : '+str(valid_nll_loss/len(valid_data._src_insts)))
		print('NLL_X_Loss : '+str(valid_nll_loss_x/len(valid_data._src_insts)))
		print('KL_Loss : '+str(valid_kl_loss/len(valid_data._src_insts)))
		print('CPC_Loss : '+str(valid_cpc_loss/len(valid_data._src_insts)))

		# torch.save(all_predictions,path.join(opt.model_name,'epochs','valid_preds'+str(epoch_i+1)+'.pt'))
		# torch.save(all_targets,path.join(opt.model_name,'epochs','valid_targets'+str(epoch_i+1)+'.pt'))
		valid_metrics = evals.compute_metrics(all_predictions,all_targets,0,opt,elapsed,all_metrics=True, THRESHOLDS=THRESHOLDS)
		valid_losses += [valid_loss]
  
  
		# metric = 'ebF1' # choose metric for saving best model
		# if  valid_metrics[metric] >= best_valid_metrics[metric]:
		# 	best_valid_metrics = valid_metrics

		################################## TEST ###################################
		start = time.time()
		all_predictions, all_targets, test_loss = test_epoch(model, test_data,opt,'(Testing)')
		elapsed = ((time.time()-start)/60)
		print('\n(Testing) elapse: {elapse:3.3f} min'.format(elapse=elapsed))
		test_total_loss, test_nll_loss, test_nll_loss_x, test_kl_loss, test_cpc_loss = test_loss
		print('Total_Loss : '+str(test_total_loss/len(test_data._src_insts)))
		print('NLL_Loss : '+str(test_nll_loss/len(test_data._src_insts)))
		print('NLL_X_Loss : '+str(test_nll_loss_x/len(test_data._src_insts)))
		print('KL_Loss : '+str(test_kl_loss/len(test_data._src_insts)))
		print('CPC_Loss : '+str(test_cpc_loss/len(test_data._src_insts)))

		# torch.save(all_predictions,path.join(opt.model_name,'epochs','test_preds'+str(epoch_i+1)+'.pt'))
		# torch.save(all_targets,path.join(opt.model_name,'epochs','test_targets'+str(epoch_i+1)+'.pt'))
		test_metrics = evals.compute_metrics(all_predictions,all_targets,0,opt,elapsed,all_metrics=True, THRESHOLDS=THRESHOLDS)
		
		best_valid,best_test = logger.evaluate(train_metrics,valid_metrics,test_metrics,epoch_i,opt.total_num_parameters)
		print('\n')
		print('**********************************')
		print('best ACC:  '+str(best_test['ACC']))
		print('best HA:   '+str(best_test['HA']))
		print('best ebF1: '+str(best_test['ebF1']))
		print('best miF1: '+str(best_test['miF1']))
		print('best maF1: '+str(best_test['maF1']))
		print('best meanAUC:  '+str(best_test['meanAUC']))
		print('best meanAUPR: '+str(best_test['meanAUPR']))
		print('best meanFDR: '+str(best_test['meanFDR']))
		print('**********************************')
		print(opt.model_name)

		losses.append([epoch_i+1,train_loss,valid_loss,test_loss])
		
		if not 'test' in opt.model_name and not opt.test_only:
			utils.save_model(opt,epoch_i,model,valid_loss,valid_losses)

		loss_file.write(str(int(epoch_i+1)))
		loss_file.write(','+str(train_loss))
		loss_file.write(','+str(valid_loss))
		loss_file.write(','+str(test_loss))
		loss_file.write('\n')

		if best_test['ebF1'] >= max_metrics['ebF1']:
			max_metrics = best_test

		if trial is not None:
			trial.report(max_metrics['ebF1'], step=epoch_i)  # val_metric
			if trial.should_prune():
				raise optuna.TrialPruned()

	return max_metrics
	
