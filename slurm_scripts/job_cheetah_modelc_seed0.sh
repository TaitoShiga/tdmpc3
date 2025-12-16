#!/bin/bash
#SBATCH -J cheetah-modelc-seed0
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

# 実験コマンド: Cheetah Model C (Proposed Method), Seed 0
python tdmpc2/train.py task=cheetah-run_randomized use_model_c=true seed=0 steps=1000000 \
    c_phys_dim=1 phys_param_type=friction phys_param_normalization=standard \
    context_length=50 gru_hidden_dim=256 \
    exp_name=cheetah_modelc eval_freq=500 compile=false enable_wandb=false \
    save_video=false

