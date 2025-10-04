import os.path as path
import os
import torch
# python main.py -dataset reuters -batch_size 64 -d_model 512 -d_inner_hid 1024 -n_layers_sample_enc 3 -n_layers_label_enc 3 -n_head 8 -epoch 200 -gpu_id 0 -sample_enc_dropout 0.2 -label_enc_dropout 0.2 -lr 0.0002 -decoder_type 'simple' -enc_transform 'special_token' -special_token_init 'normal' 

# python main.py -dataset reuters -batch_size 64 -d_model 512 -d_inner_hid 1024 -n_layers_sample_enc 3 -n_layers_label_enc 3 -n_head 8 -epoch 200 -gpu_id 0 -sample_enc_dropout 0.2 -label_enc_dropout 0.0 -lr 0.0002 -decoder_type 'simple' -enc_transform 'special_token' -special_token_init 'normal' -feature_aggregate 'simple_attention'

# python main.py -dataset reuters -batch_size 64 -d_model 512 -d_inner_hid 1024 -n_layers_sample_enc 3 -n_layers_label_enc 3 -n_head 8 -epoch 200 -gpu_id 0 -sample_enc_dropout 0.2 -label_enc_dropout 0.0 -lr 0.0002 -decoder_type 'simple' -enc_transform 'special_token' -special_token_init 'normal' -feature_aggregate 'self_attention'

# python main.py -dataset reuters -batch_size 64 -d_model 512 -d_inner_hid 1024 -n_layers_sample_enc 3 -n_layers_label_enc 3 -n_head 8 -epoch 200 -gpu_id 0 -sample_enc_dropout 0.2 -label_enc_dropout 0.0 -lr 0.0002 -decoder_type 'simple' -enc_transform 'special_token' -special_token_init 'normal' -node2hyperedge_aggregate 'simple_attention'

# python main.py -dataset reuters -batch_size 64 -d_model 512 -d_inner_hid 1024 -n_layers_sample_enc 3 -n_layers_label_enc 3 -n_head 8 -epoch 200 -gpu_id 1 -sample_enc_dropout 0.2 -label_enc_dropout 0.0 -lr 0.0002 -decoder_type 'simple' -enc_transform 'special_token' -special_token_init 'normal' -node2hyperedge_aggregate 'soft_attention'

# python main.py -dataset reuters -batch_size 64 -d_model 512 -d_inner_hid 1024 -n_layers_sample_enc 3 -n_layers_label_enc 3 -n_head 8 -epoch 200 -gpu_id 1 -sample_enc_dropout 0.2 -label_enc_dropout 0.0 -lr 0.0002 -decoder_type 'simple' -enc_transform 'special_token' -special_token_init 'normal' -node_update 'gated'

# python main.py -dataset reuters -batch_size 64 -d_model 512 -d_inner_hid 1024 -n_layers_sample_enc 3 -n_layers_label_enc 3 -n_head 8 -epoch 200 -gpu_id 1 -sample_enc_dropout 0.2 -label_enc_dropout 0.0 -lr 0.0002 -decoder_type 'simple' -enc_transform 'special_token' -special_token_init 'normal' -feature_aggregate 'self_attention' -node2hyperedge_aggregate 'soft_attention' -node_update 'gated'

def get_args(parser):
    parser.add_argument('-dataroot', type=str, default='data/')
    parser.add_argument('-dataset', type=str, default='reuters')
    parser.add_argument('-results_dir', type=str, default='results/')
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-test_batch_size', type=int, default=-1)
    parser.add_argument('-d_model', type=int, default=512)  # model dimension
    parser.add_argument('-d_emb', type=int, default=512)  # embedding dimension
    parser.add_argument('-d_latent', type=int, default=64)  # latent dimension
    parser.add_argument('-d_inner_hid', type=int, default=-1)  # model hidden dimension
    parser.add_argument('-d_k', type=int, default=-1)
    parser.add_argument('-d_v', type=int, default=-1)
    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers_sample_enc', type=int, default=5)
    parser.add_argument('-n_layers_label_enc', type=int, default=5)
    parser.add_argument('-lr', type=float, default=0.0002)
    parser.add_argument('-lr_step_size', type=int, default=10)
    parser.add_argument('-lr_decay', type=float, default=0.8)
    parser.add_argument('-sample_enc_dropout', type=float, default=0.1)
    parser.add_argument('-label_enc_dropout', type=float, default=0.1)
    parser.add_argument('-encoder_type', type=str, choices=['MLP', 'DeepSets', 'SetTransformer'], default='MLP')
    parser.add_argument('-enc_transform', type=str, choices=['max', 'mean', 'special_token'], default='special_token')
    parser.add_argument('-special_token_init', type=str, choices=['random', 'xavier_uniform', 'xavier_normal',
                                                                  'kaiming_uniform', 'kaiming_normal', 'normal'],
                        default='normal')
    parser.add_argument('-feature_aggregate', type=str, choices=['mean', 'max', 'simple_attention',
                                                                 'self_attention'], default='mean')
    parser.add_argument('-node2hyperedge_aggregate', type=str, choices=['mean', 'max', 'simple_attention',
                                                                 'soft_attention'], default='mean')
    parser.add_argument('-node_update', type=str, choices=['simple', 'gated'], default='simple')
    parser.add_argument('-br_threshold', type=float, default=0.5)
    parser.add_argument('-feat_mode', type=str, choices=['multi-hot', 'tokens', 'vec'], default='tokens')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-multi_gpu', action='store_true')
    parser.add_argument('-viz', action='store_true')
    parser.add_argument('-gpu_id', type=int, default=-1)
    parser.add_argument('-no_enc_pos_embedding', action='store_true')
    parser.add_argument('-test_only', action='store_true')
    parser.add_argument('-load_pretrained', action='store_true')
    parser.add_argument('-nll_coeff', type=float, default=1.0)
    opt = parser.parse_args()
    return opt


def config_args(opt):
    opt.multi_gpu = False

    # if 'reuters' in opt.dataset or 'bibtext' in opt.dataset:

    if opt.dataset in ['deepsea', 'gm12878', 'gm12878_unique2', 'gm12878_unique', 'tcell']:
        opt.feat_mode = 'onehot'

    if opt.d_v == -1:
        opt.d_v = int(opt.d_model / opt.n_head)
    if opt.d_k == -1:
        opt.d_k = int(opt.d_model / opt.n_head)

    if opt.dataset in ['yeast', 'scene', 'nuswide_vector']:
        opt.feat_mode = 'vec'
        opt.no_enc_pos_embedding = True

    if opt.d_inner_hid == -1:
        opt.d_inner_hid = int(opt.d_model * 2)

    opt.model_name = ''

    opt.model_name += 'enc_trans_' + opt.enc_transform

    opt.model_name += '.token_init_' + opt.special_token_init

    opt.model_name += '.feat_agg_' + opt.feature_aggregate

    opt.model_name += '.n2he_agg_' + opt.node2hyperedge_aggregate

    opt.model_name += '.n_update_' + opt.node_update

    opt.model_name += '.enc_' + opt.encoder_type
    opt.model_name += '.' + str(opt.d_model)
    opt.model_name += '.' + str(opt.d_inner_hid)
    opt.model_name += '.' + str(opt.d_k)
    opt.model_name += '.' + str(opt.d_v)
    opt.model_name += '.nlayers_' + str(opt.n_layers_sample_enc) + '_' + str(opt.n_layers_label_enc)
    opt.model_name += '.nheads_' + str(opt.n_head)

    opt.model_name += '.bsz_' + str(opt.batch_size)

    if opt.test_batch_size <= 0:
        opt.test_batch_size = opt.batch_size

    opt.model_name += '.test_bsz_' + str(opt.batch_size)

    opt.model_name += '.lr_' + str(opt.lr).split('.')[1]
    if opt.lr_decay > 0:
        opt.model_name += '.decay_' + str(opt.lr_decay).replace('.', '') + '_' + str(opt.lr_step_size)

    opt.model_name += '.sample_drop_' + ("%.2f" % opt.sample_enc_dropout).split('.')[1]
    opt.model_name += '.label_drop_' + ("%.2f" % opt.label_enc_dropout).split('.')[1]

    opt.model_name += f'.br_th_{opt.br_threshold}'

    opt.model_name+= f'.epoch_{opt.epoch}'

    opt.model_name = path.join(opt.results_dir, opt.dataset, opt.model_name)

    opt.data_type = opt.dataset

    opt.dataset = path.join(opt.dataroot, opt.dataset)

    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    opt.data = path.join(opt.dataset, 'train_valid_test.pt')

    if torch.cuda.is_available() and opt.cuda:
        if opt.gpu_id != -1:
            opt.device = torch.device(f"cuda:{opt.gpu_id}")
        else:
            opt.device = torch.device("cuda")

    print(opt.model_name)

    return opt
