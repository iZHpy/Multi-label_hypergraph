python main.py -dataset yeast -batch_size 32 -d_model 512 -d_inner_hid 256 -n_head 4 -n_layers_sample_enc 2 -n_layers_label_enc 2 \
-epoch 100 -sample_enc_dropout 0.4 -label_enc_dropout 0.3 -lr 0.0001 -feature_aggregate self_attention -eta_min 5e-06 \
-T0 1 -T_mult 2 -nll_e_weight 3.5 -nll_x_weight 8.0 -kl_weight 0.4 -cpc_weight 1.0 -feat_mode vec -decoder_type MLP