#!/bin/bash
#SBATCH -J walker-model-c
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

# Model C 訓練（5 seeds）
# ランダム質量 + GRU推定器で torso_mass を推定
for seed in 0 1 2 3 4; do
    echo "Training Model C seed=${seed}..."
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
        seed=${seed} \
        save_video=false \
        enable_wandb=false \
        compile=false
done

echo "Model C training completed!"

