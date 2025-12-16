#!/bin/bash
#SBATCH -J tdmpc2-eval-seed3
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

# 評価コマンド: Seed 3  
# 4モデル（baseline, dr, c, o）× 5 params × 30 episodes
python evaluate/evaluate_all_models.py \
    --seeds 3 \
    --episodes 30 \
    --test-params 0.2 0.3 0.4 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0\
    --output results_seed3.csv \
    --logs-dir logs \
    --task pendulum-swingup \
    --model-size 5

echo "Evaluation for seed 3 completed!"

