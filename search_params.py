import argparse, math, time, warnings, copy, numpy as np, os.path as path
import utils.evals as evals
import utils.utils as utils
from utils.data_loader import process_data
import torch, torch.nn as nn, torch.nn.functional as F
import Hyperlabel.Constants as Constants
from Hyperlabel.Models import Hyperlabel
from Hyperlabel.Translator import translate
from config_args import config_args, get_args
from pdb import set_trace as stop
from tqdm import tqdm
from runner import run_model
import sys
import os
import optuna


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
args = get_args(parser)
opt = config_args(args)

def generate_train_labels(list_of_lists):
    """
    Modify a list of lists by removing elements 0, 1, 2, 3 from each sublist,
    and subtracting 3 from the remaining non-negative integers.
    """
    # Create a new list to store the modified sublists
    train_labels = []

    # Iterate over each sublist in the input list
    for sublist in list_of_lists:
        # Remove elements 0, 1, 2, 3 from the sublist
        filtered_sublist = [x for x in sublist if x not in (0, 1, 2, 3)]
        # Subtract 3 from the remaining elements
        updated_sublist = [x - 4 for x in filtered_sublist]
        # Append the updated sublist to the modified list
        train_labels.append(updated_sublist)

    # Return the modified list of lists
    return train_labels


class Logger(object):
    def __init__(self, filename="default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 确保实时写入文件

    def flush(self):
        pass


def start_logging(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    sys.stdout = Logger(log_file_path)


def train_eval(params, trial, opt):
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    
    # ========= Loading Dataset =========#
    data = torch.load(opt.data)
    if opt.dataset in ['data/reuters', 'data/bibtext', 'data/bookmarks', 'data/delicious', 'data/sider', 'data/yeast', 'data/sider', 'data/nuswide_vector', 'data/scene']:
        train_labels = generate_train_labels(data['train']['tgt'])

    opt.batch_size = params['batch_size']

    train_data, valid_data, test_data, opt = process_data(data, opt)
    model = Hyperlabel(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len_e,
        train_labels=train_labels,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=params['d_model'],
        d_word_vec=params['d_model'],
        d_emb=params['d_model'],
        d_inner_hid=params['d_inner_hid'],
        d_latent=params['d_model'],
        n_layers_sample_enc=params['n_layers_sample_enc'],
        n_layers_label_enc=params['n_layers_label_enc'],
        n_head=params['n_head'],
        sample_enc_dropout=params['sample_enc_dropout'],
        label_enc_dropout = params['label_enc_dropout'],
        encoder_type=opt.encoder_type,
        enc_transform=opt.enc_transform,
        special_token_init=opt.special_token_init,
        feature_aggregate=opt.feature_aggregate,
        node2hyperedge_aggregate = opt.node2hyperedge_aggregate,
        node_update=opt.node_update,
        feat_mode=opt.feat_mode,
        no_enc_pos_embedding=opt.no_enc_pos_embedding)

    opt.total_num_parameters = int(utils.count_parameters(model))
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=params['lr'], betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=params['eta_min'], T_0=params['T0'] * len(train_data), T_mult=params['T_mult'])
    crit = None
    adv_optimizer = None
    if torch.cuda.is_available() and opt.cuda:
        model = model.to(opt.device)

    metrics = run_model(model, train_data, valid_data, test_data, crit, optimizer, adv_optimizer, scheduler, opt, trial=trial)
    if trial is not None and isinstance(metrics, dict):
        trial.set_user_attr("metrics", {k: v for k, v in metrics.items()})
    return metrics['ebF1']


def objective(trial):
    params = {
        'd_model': trial.suggest_categorical('d_model', [64, 128, 256, 512]),
        'd_inner_hid': trial.suggest_categorical('d_inner_hid', [256, 512, 1024, 2048]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'n_layers_sample_enc': trial.suggest_int('n_layers_sample_enc', 2, 3, 5),
        'n_layers_label_enc': trial.suggest_int('n_layers_label_enc', 2, 3, 5),
        'n_head': trial.suggest_categorical('n_head', [4, 8]),
        'sample_enc_dropout': trial.suggest_float("sample_enc_dropout", 0.0, 0.5, step=0.1),
        'label_enc_dropout': trial.suggest_float('label_enc_dropout', 0.0, 0.5, step=0.1),
        'lr': trial.suggest_categorical('lr', [2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 8e-4, 1e-3]),
        'eta_min': trial.suggest_categorical('eta_min', [1e-6, 5e-6, 1e-5]),
        'T0': trial.suggest_int('T0', 1, 2, 4),
        'T_mult': trial.suggest_categorical('T_mult', [2, 4]),
        'nll_e_weight': trial.suggest_float('nll_e_weight', 1.0, 6.0, step=1),
        'nll_x_weight': trial.suggest_float('nll_x_weight', 1.0, 8.0, step=1),
        'kl_weight': trial.suggest_float('kl_weight', 0, 1, step=0.2),
        'cpc_weight': trial.suggest_float('cpc_weight', 0, 1, step=0.2),
    }

    # params = {
    #     'd_model': trial.suggest_categorical('d_model', [64, 128, 256, 512]),
    #     'd_inner_hid': trial.suggest_categorical('d_inner_hid', [256, 512, 1024, 2048]),
    #     'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
    #     'n_layers_sample_enc': trial.suggest_int('n_layers_sample_enc', 2, 3, 5),
    #     'n_layers_label_enc': trial.suggest_int('n_layers_label_enc', 2, 3, 5),
    #     'n_head': trial.suggest_categorical('n_head', [4, 8]),
    #     'sample_enc_dropout': trial.suggest_float('sample_enc_dropout', 0.0, 0.5),
    #     'label_enc_dropout': trial.suggest_float('label_enc_dropout', 0.0, 0.5),
    #     'lr': trial.suggest_float('lr', 2e-5, 1e-3, log=True),
    #     'eta_min': trial.suggest_float('eta_min', 1e-6, 1e-5, log=True),
    #     'T0': trial.suggest_int('T0', 1, 2, 4),
    #     'T_mult': trial.suggest_categorical('T_mult', [2, 4]),
    #     'nll_e_weight': trial.suggest_float('nll_e_weight', 1.0, 6.0),
    #     'nll_x_weight': trial.suggest_float('nll_x_weight', 1.0, 8.0),
    #     'kl_weight': trial.suggest_float('kl_weight', 0, 1),
    #     'cpc_weight': trial.suggest_float('cpc_weight', 0, 1),
    # }

    metric = train_eval(params, opt=opt, trial=trial)
    return metric

if __name__ == '__main__':
    log_file_path = os.path.join(opt.model_search, 'search.txt')
    start_logging(log_file_path)
    pruner = optuna.pruners.MedianPruner(  # 中度剪枝
        n_startup_trials=5,      # 前5个不剪
        n_warmup_steps=0,        # 不等warmup步也行（你用epoch作为step）
        interval_steps=1         # 每个epoch都评估
    )
    study = optuna.create_study(direction='maximize', pruner=None)
    study.optimize(objective, n_trials=35, show_progress_bar=True)
    
    for t in study.trials:
        print(f"Trial {t.number} | state={t.state} | value={t.value}")
        if "metrics" in t.user_attrs:
            print("  metrics:", t.user_attrs["metrics"])
    
    print('Best trial:')
    trial = study.best_trial
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
