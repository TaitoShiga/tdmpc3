#!/bin/bash
#SBATCH -J tdmpc2-test-eval
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# === conda 初期化（これが重要）===
eval "$(conda shell.bash hook)"
conda activate tdmpc2

# ログディレクトリ作成
mkdir -p logs

# 作業ディレクトリへ移動
cd ~/tdmpc3/tdmpc3

echo "=============================================="
echo "Test evaluation job starting..."
echo "=============================================="
echo ""

# テスト評価: 1 seed × 1 param × 5 episodes
# これが成功したら本番のジョブを投入する
python evaluate/evaluate_all_models.py \
    --seeds 0 \
    --episodes 5 \
    --test-params 1.0 \
    --output results_test.csv \
    --logs-dir logs \
    --task pendulum-swingup \
    --model-size 5

echo ""
echo "=============================================="
echo "Test evaluation completed!"
echo "=============================================="
echo ""
echo "Check results:"
echo "  cat results_test.csv"
echo ""
echo "If successful, submit full evaluation:"
echo "  bash slurm_scripts/submit_eval_all.sh"

