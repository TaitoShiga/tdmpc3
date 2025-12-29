#!/bin/bash
#SBATCH -J hopper-model-c
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

# Model C 訓練（5 seeds）
# ランダム長 + GRU推定器で thigh_length を推定
for seed in 0 1 2 3 4; do
    echo "Training Model C seed=${seed}..."
    python tdmpc2/train.py \
        task=hopper-hop_backwards_randomized \
        exp_name=hopper_model_c \
        use_model_c=true \
        c_phys_dim=1 \
        phys_param_type=length \
        phys_param_normalization=standard \
        context_length=50 \
        gru_hidden_dim=256 \
        steps=500000 \
        seed=${seed} \
        save_video=false \
        enable_wandb=false \
        compile=false
done

echo "Model C training completed!"

