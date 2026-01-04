#!/bin/bash
#SBATCH -J walker-oracle-seed1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# === conda 初期化（これが重要）===
eval "$(conda shell.bash hook)"
conda activate tdmpc2

# ログディレクトリ作成
mkdir -p logs

# 作業ディレクトリへ移動
cd ~/tdmpc3/tdmpc3

echo "Training Oracle seed=1..."
python tdmpc2/train.py \
    task=walker-walk_randomized \
    exp_name=walker_oracle \
    use_oracle=true \
    c_phys_dim=1 \
    phys_param_type=mass \
    phys_param_normalization=standard \
    steps=100000 \
    eval_freq=500 \
    seed=1 \
    save_video=false \
    enable_wandb=false \
    compile=false

echo "Oracle seed=1 completed!"

