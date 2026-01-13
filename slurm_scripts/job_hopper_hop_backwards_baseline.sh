#!/bin/bash
#SBATCH -J hopper-hop-backwards-baseline
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 3-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# === conda 初期化（これが重要）===
eval "$(conda shell.bash hook)"
conda activate tdmpc2

# ログディレクトリ作成
mkdir -p logs

# 作業ディレクトリへ移動
cd ~/tdmpc3/tdmpc3

echo "Training Hopper Hop Backwards Baseline..."
python tdmpc2/train.py \
    task=hopper-hop-backwards \
    exp_name=hopper_hop_backwards_baseline \
    steps=500000 \
    eval_freq=500 \
    seed=0 \
    save_video=false \
    enable_wandb=false \
    compile=false

echo "Hopper Hop Backwards Baseline completed!"
