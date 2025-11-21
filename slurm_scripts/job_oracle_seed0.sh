#!/bin/bash
#SBATCH -J tdmpc2-oracle-seed0
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# === conda 初期化（これが重要）===
eval "$(conda shell.bash hook)"
conda activate tdmpc2

# ログディレクトリ作成
mkdir -p logs

# 作業ディレクトリへ移動
cd ~/tdmpc3/tdmpc3

# 実験コマンド: Model O (Oracle), Seed 0
python tdmpc2/train.py task=pendulum-swingup-randomized use_oracle=true seed=0 steps=100000 \
    exp_name=oracle log_interval=100 compile=false enable_wandb=false

