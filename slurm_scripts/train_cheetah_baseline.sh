#!/bin/bash
#SBATCH -J cheetah-baseline
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# === conda 初期化 ===
eval "$(conda shell.bash hook)"
conda activate tdmpc2

# ログディレクトリ作成
mkdir -p logs

# 作業ディレクトリへ移動
cd ~/Research/tdmpc2-2/tdmpc2

# Baseline 訓練（5 seeds）
for seed in 0 1 2 3 4; do
    echo "Training Baseline seed=${seed}..."
    python tdmpc2/train.py \
        task=cheetah-run_friction04 \
        exp_name=cheetah_baseline \
        steps=1000000 \
        seed=${seed} \
        save_video=false \
        enable_wandb=false \
        compile=false
done

echo "Baseline training completed!"

