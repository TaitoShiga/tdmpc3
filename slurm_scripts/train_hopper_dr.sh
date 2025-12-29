#!/bin/bash
#SBATCH -J hopper-dr
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
cd ~/tdmpc3/tdmpc3

# DR 訓練（5 seeds）
# ランダム長 thigh_length ~ uniform(0.25, 0.45)
for seed in 0 1 2 3 4; do
    echo "Training DR seed=${seed}..."
    python tdmpc2/train.py \
        task=hopper-hop_backwards_randomized \
        exp_name=hopper_dr \
        steps=500000 \
        seed=${seed} \
        save_video=false \
        enable_wandb=false \
        compile=false eval_freq=500
done

echo "DR training completed!"

