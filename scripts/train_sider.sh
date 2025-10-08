python main.py -dataset sider-batch_size 32 -d_model 64 -d_inner_hid 256 -n_head 8 -n_layers_sample_enc 2 -n_layers_label_enc 2 \
-epoch 40 -sample_enc_dropout 0.0 -label_enc_dropout 0.4 -lr 0.0002 -feature_aggregate self_attention -eta_min 1e-05 \
-T0 1 -T_mult 4 -nll_e_weight 3.5 -nll_x_weight 7.5 -kl_weight 0 -cpc_weight 0.8 -decoder_type MLP