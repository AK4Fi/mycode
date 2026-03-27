#!/bin/bash

# =====================================================================
# 恶意代码多模态检测模型：黑盒抗混淆能力一键测试脚本
# =====================================================================

# 1. 目录配置 (根据你的仓库代码自动推导)
WEIGHTS_DIR="./weights"
DATA_DIR="/root/autodl-tmp/Kaggle2015/full_data"

# 2. 定义要测试的 NOP 比例
NOP_RATIOS=(10 20 30)

echo "========================================================"
echo " 🚀 开始自动化黑盒对抗测试 (干净数据 + NOP 注入数据)"
echo "========================================================"

# ---------------------------------------------------------
# 测试一：0% NOP (干净测试集，作为性能 Baseline)
# ---------------------------------------------------------
echo -e "\n[+] 测试 0% NOP (干净测试集 test.csv)"
echo "--------------------------------------------------------"
echo "-> 传统头部截断模型 (trunc_head):"
python test_robustness.py \
    --checkpoint ${WEIGHTS_DIR}/exp_trunc_head.pth \
    --test_csv ${DATA_DIR}/test.csv \
    --truncate head_only

echo -e "\n-> 本文多模态模型 (trunc_cfg_guided):"
python test_robustness.py \
    --checkpoint ${WEIGHTS_DIR}/exp_trunc_cfg_guided.pth \
    --test_csv ${DATA_DIR}/test.csv \
    --truncate cfg_guided

# ---------------------------------------------------------
# 测试二：遍历 10%, 20%, 30% 的 NOP 污染测试集
# ---------------------------------------------------------
for ratio in "${NOP_RATIOS[@]}"; do
    echo -e "\n\n[+] 测试 ${ratio}% NOP 注入强度 (test_nop_${ratio}.csv)"
    echo "--------------------------------------------------------"
    
    echo "-> 传统头部截断模型 (trunc_head) - 预期会暴跌:"
    python test_robustness.py \
        --checkpoint ${WEIGHTS_DIR}/exp_trunc_head.pth \
        --test_csv ${DATA_DIR}/test_nop_${ratio}.csv \
        --truncate head_only
        
    echo -e "\n-> 本文多模态模型 (trunc_cfg_guided) - 预期保持坚挺:"
    python test_robustness.py \
        --checkpoint ${WEIGHTS_DIR}/exp_trunc_cfg_guided.pth \
        --test_csv ${DATA_DIR}/test_nop_${ratio}.csv \
        --truncate cfg_guided
done

echo -e "\n========================================================"
echo " [√] 所有梯度强度的对抗测试运行完毕！"
echo " 请收集上述终端输出的 Macro-F1 数据，绘制折线图填入论文。"
echo "========================================================"
