#!/bin/bash
#SBATCH -J walker-modelc-seed2
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

echo "Training Model C seed=2..."
python tdmpc2/train.py \
    task=walker-walk_randomized \
    exp_name=walker_model_c \
    use_model_c=true \
    c_phys_dim=1 \
    phys_param_type=mass \
    phys_param_normalization=standard \
    context_length=50 \
    gru_hidden_dim=256 \
    steps=100000 \
    eval_freq=500 \
    seed=2 \
    save_video=false \
    enable_wandb=false \
    compile=false

echo "Model C seed=2 completed!"

