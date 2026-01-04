#!/bin/bash
#SBATCH -J walker-actuator-eval
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# === conda 初期化 ===
# Option 1: bashrcから読み込む
source ~/.bashrc

# Option 2: conda hookを使う（上記で失敗する場合）
# eval "$(conda shell.bash hook)"

# Option 3: module systemを使う（クラスタ環境によっては）
# module load anaconda3

conda activate tdmpc2
mkdir -p logs results

# デフォルト設定（利用可能なseedのみ使用）
# Baseline: seed0, DR: seed3, Model C: seed0, Oracle: seed0
OUTPUT="${OUTPUT:-results_walker_actuator.csv}"
EPISODES="${EPISODES:-30}"
MODEL_SIZE="${MODEL_SIZE:-5}"
LOGS_DIR="${LOGS_DIR:-logs}"
IN_DIST_ONLY="${IN_DIST_ONLY:-false}"

echo "Evaluating Walker actuator models..."
echo "  Episodes per (model, seed, scale): ${EPISODES}"
echo "  Logs dir: ${LOGS_DIR}"
echo "  Output: ${OUTPUT}"
echo "  In-Dist only: ${IN_DIST_ONLY}"
echo ""
echo "Using available seeds per model:"
echo "  - Baseline: seed0"
echo "  - DR: seed3"
echo "  - Model C: seed0"
echo "  - Oracle: seed0"
echo ""

# スケール範囲を構築
if [ "${IN_DIST_ONLY}" = "true" ]; then
    SCALES="0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4"
    echo "Evaluating In-Distribution scales only (0.4-1.4)"
else
    # In-Dist + OOD
    SCALES="0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7"
    echo "Evaluating In-Distribution (0.4-1.4) + OOD (0.2, 0.3, 1.5, 1.6, 1.7)"
fi
echo ""

python evaluate/evaluate_walker_actuator_all_models.py \
    --episodes ${EPISODES} \
    --actuator-scales ${SCALES} \
    --output ${OUTPUT} \
    --logs-dir ${LOGS_DIR} \
    --model-size ${MODEL_SIZE}

echo "Walker actuator evaluation completed."
