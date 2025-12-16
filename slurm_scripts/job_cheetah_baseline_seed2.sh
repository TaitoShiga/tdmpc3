#!/bin/bash
#SBATCH -J cheetah-baseline-seed2
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

# 実験コマンド: Cheetah Baseline, Seed 2
python tdmpc2/train.py task=cheetah-run_friction04 seed=2 steps=1000000 \
    exp_name=cheetah_baseline eval_freq=500 compile=false enable_wandb=false \
    save_video=false

