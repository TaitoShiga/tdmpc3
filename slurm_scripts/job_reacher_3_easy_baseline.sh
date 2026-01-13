#!/bin/bash
#SBATCH -J reacher-3-easy-baseline
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# === conda 初期化（これが重要）===
eval "$(conda shell.bash hook)"
conda activate tdmpc2

# ログディレクトリ作成
mkdir -p logs

# 作業ディレクトリへ移動
cd ~/tdmpc3/tdmpc3

echo "Training Reacher 3 Easy Baseline..."
python tdmpc2/train.py \
    task=reacher-3-easy \
    exp_name=reacher_3_easy_baseline \
    steps=200000 \
    eval_freq=500 \
    seed=0 \
    save_video=false \
    enable_wandb=false \
    compile=false

echo "Reacher 3 Easy Baseline completed!"
