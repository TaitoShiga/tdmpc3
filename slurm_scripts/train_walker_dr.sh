#!/bin/bash
#SBATCH -J walker-dr
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
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
# ランダム質量 torso_mass ~ [0.5×, 2.5×]
for seed in 0 1 2 3 4; do
    echo "Training DR seed=${seed}..."
    python tdmpc2/train.py \
        task=walker-walk_randomized \
        exp_name=walker_dr \
        steps=100000 \
        eval_freq=500 \
        seed=${seed} \
        save_video=false \
        enable_wandb=false \
        compile=false
done

echo "DR training completed!"

