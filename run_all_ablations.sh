#!/bin/bash

# ==================================
# 实验一：序列特征降维消融
# ==================================
python train_ablation.py --exp_name trunc_head --truncate head_only
python train_ablation.py --exp_name trunc_head_tail --truncate head_tail
python train_ablation.py --exp_name trunc_entropy --truncate entropy
python train_ablation.py --exp_name trunc_cfg_guided --truncate cfg_guided

# ==================================
# 实验二：模态有效性消融
# ==================================
python train_ablation.py --exp_name mod_cfg_only --modality cfg_only --truncate entropy
python train_ablation.py --exp_name mod_opc_only --modality opcode_only --truncate entropy

# ==================================
# 实验三：融合机制与方向消融
# ==================================
python train_ablation.py --exp_name fuse_concat --fusion concat --truncate entropy
python train_ablation.py --exp_name fuse_dense --fusion dense_concat --truncate entropy
python train_ablation.py --exp_name fuse_seq2cfg --attn_dir seq2cfg --truncate entropy

echo "done"