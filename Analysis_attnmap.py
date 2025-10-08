# -*- coding: utf-8 -*-
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

# ========== 路径 ==========
# ATTN_PATH = './attn_map/epoch20_idx1984.pt'
ATTN_PATH = './attn_map/epoch22_idx1925.pt'
DICT_PATH = './data/reuters/train_valid_test.pt'

# ========== 读取 ==========
d = torch.load(ATTN_PATH, map_location='cpu')
src_raw = d['src_raw']         # 一条样本的 source token id 列表，长度应为 302
tgt_raw = d['tgt_raw']         # 一条样本的 target label id 列表，长度应为 90
attn_e  = d['attn_e']          # 形状 [90, 1]
attn_x  = d['attn_x']          # 形状 [90, 302]
eps = 1e-8
row_sum = attn_x.sum(dim=0, keepdim=True).clamp_min(eps)
x_min = attn_x.min(dim=1, keepdim=True).values
x_max = attn_x.max(dim=1, keepdim=True).values
attn_x = (attn_x - x_min) / (x_max - x_min + eps)


data = torch.load(DICT_PATH, map_location='cpu')
src2id = data['dict']['src']
id2src = {v: k for k, v in src2id.items()}
tgt2id = data['dict']['tgt']
id2tgt = {v: k for k, v in tgt2id.items()}

# ========== 转 numpy ==========
attn_e_np = attn_e.detach().cpu().numpy().reshape(-1)
attn_x_np = attn_x.detach().cpu().numpy()

# ========== 构造坐标轴文字 ==========
def safe_text(txt, max_len=20):
    s = str(txt)
    return s if len(s) <= max_len else (s[:max_len-1] + '…')

import random
src_idx = [1,2,3,4,5,6,8,9,11,12,13,14,151,16,18,19,20,21,23,24,25]
pos_idx = torch.nonzero(tgt_raw, as_tuple=True)[0].tolist()
neg_idx = torch.nonzero(tgt_raw == 0, as_tuple=True)[0].tolist()
sample_neg =  random.sample(neg_idx, k=5)
tgt_idx = pos_idx + sample_neg
attn_x_np = attn_x_np[tgt_idx]
attn_x_np = attn_x_np[:, src_idx]
attn_e_np = attn_e_np[tgt_idx]
label_names = []
for lid in tgt_idx:
    name = id2tgt.get(int(lid) + 4, f'<UNK:{lid}>')
    label_names.append(safe_text(name, 24))


token_texts = []
for id in src_idx:
    tid = src_raw[id]
    tok = id2src.get(int(tid), f'<UNK:{tid}>')
    token_texts.append(safe_text(tok, 18))
# for tid in src_raw[src_idx]:
#     tok = id2src.get(int(tid), f'<UNK:{tid}>')
#     token_texts.append(safe_text(tok, 18))

# ========== 字体参数 ==========
FONT_TITLE = 40
FONT_LABEL = 35
FONT_TICK  = 30
FONT_CB    = 22

# ========== 画 attn_x 热力图 ==========
plt.figure(figsize=(20, 10))
im = plt.imshow(attn_x_np, aspect='auto', interpolation='nearest')
cbar = plt.colorbar(im, fraction=0.025, pad=0.02)
cbar.ax.tick_params(labelsize=FONT_CB)

plt.title('Feature-Label Correlation', fontsize=FONT_TITLE)

def sparse_ticks(n, step_target=20):
    step = max(1, int(round(n / step_target)))
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return idxs

yticks = sparse_ticks(len(label_names), step_target=15)
xticks = sparse_ticks(len(token_texts), step_target=25)

plt.yticks(yticks, [label_names[i] for i in yticks], fontsize=FONT_TICK)
plt.xticks(xticks, [token_texts[i] for i in xticks], rotation=60, ha='right', fontsize=FONT_TICK)

plt.xlabel('Features', fontsize=FONT_LABEL)
plt.ylabel('Labels', fontsize=FONT_LABEL)
plt.tight_layout()
plt.savefig('attn_x_heatmap.pdf', dpi=300)

# ========== 画 attn_e 柱状图 ==========
plt.figure(figsize=(18, 6))
x = np.arange(len(label_names))
plt.bar(x, attn_e_np, color='skyblue')
plt.title('Feature-Label Correlation', fontsize=FONT_TITLE)
plt.xlabel('Features', fontsize=FONT_LABEL)
plt.ylabel('Labels', fontsize=FONT_LABEL)

xticks2 = sparse_ticks(len(label_names), step_target=25)
plt.xticks(xticks2, [label_names[i] for i in xticks2], rotation=60, ha='right', fontsize=FONT_TICK)
plt.yticks(fontsize=FONT_TICK)

plt.tight_layout()
plt.savefig('attn_e_bars.png', dpi=1200)
plt.show()

print('Saved: attn_x_heatmap.png, attn_e_bars.png')
