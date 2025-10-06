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


def main(opt):
    np.random.seed(opt.seed) 
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    
    log_file_path = os.path.join(opt.model_name, 'results.txt')
    start_logging(log_file_path)

    # ========= Loading Dataset =========#
    data = torch.load(opt.data)

    if opt.dataset in ['data/reuters', 'data/bibtext', 'data/bookmarks', 'data/delicious', 'data/sider', 'data/yeast', 'data/sider', 'data/nuswide_vector', 'data/scene']:
        train_labels = generate_train_labels(data['train']['tgt'])

    train_data, valid_data, test_data, opt = process_data(data, opt)
 
    print(opt)

    # ========= Preparing Model =========#
    model = Hyperlabel(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len_e,
        train_labels=train_labels,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_emb=opt.d_emb,
        d_inner_hid=opt.d_inner_hid,
        d_latent=opt.d_latent,
        n_layers_sample_enc=opt.n_layers_sample_enc,
        n_layers_label_enc=opt.n_layers_label_enc,
        n_head=opt.n_head,
        sample_enc_dropout=opt.sample_enc_dropout,
        label_enc_dropout = opt.label_enc_dropout,
        encoder_type=opt.encoder_type,
        enc_transform=opt.enc_transform,
        special_token_init=opt.special_token_init,
        feature_aggregate=opt.feature_aggregate,
        node2hyperedge_aggregate = opt.node2hyperedge_aggregate,
        node_update=opt.node_update,
        feat_mode=opt.feat_mode,
        no_enc_pos_embedding=opt.no_enc_pos_embedding)

    print(model)
    print(opt.model_name)

    opt.total_num_parameters = int(utils.count_parameters(model))

    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    
    # totol_steps = len(train_data) * opt.epoch
    # warm_steps = int(totol_steps * 0.05)
    # scheduler = utils.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warm_steps, num_training_steps=totol_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=opt.eta_min, T_0=opt.T0 * len(train_data), T_mult=opt.T_mult)
    # scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.lr_decay, last_epoch=-1)

    adv_optimizer = None

    crit = utils.get_criterion(opt)

    # if torch.cuda.device_count() > 1 and opt.multi_gpu:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    if torch.cuda.is_available() and opt.cuda:
        model = model.to(opt.device)

        crit = crit.to(opt.device)

    if opt.load_pretrained:
        checkpoint = torch.load(opt.model_name + '/model.chkpt')
        model.load_state_dict(checkpoint['model'])

    try:
        run_model(model, train_data, valid_data, test_data, crit, optimizer, adv_optimizer, scheduler, opt,
                  data['dict'])
    except KeyboardInterrupt:
        print('-' * 89 + '\nManual Exit')
        exit()

    sys.stdout = sys.stdout.terminal


if __name__ == '__main__':
    main(opt)
