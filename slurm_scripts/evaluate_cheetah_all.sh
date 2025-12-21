#!/bin/bash
#SBATCH -J cheetah-eval
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

# 作業ディレクトリへ移動（正しいパス）
cd ~/tdmpc3/tdmpc3

# 4モデル × 5 seeds × 4 frictions の評価
echo "Starting Cheetah evaluation..."
python scripts/evaluate_cheetah_all_models.py

echo "Cheetah evaluation completed!"
echo "Results saved to: cheetah_evaluation_results.csv"

