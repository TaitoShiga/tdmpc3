#!/bin/bash
#SBATCH -J walker-actuator-eval
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

eval "$(conda shell.bash hook)"
conda activate tdmpc2
mkdir -p logs
cd ~/tdmpc3/tdmpc3

SEEDS="${SEEDS:-0 1 2 3 4}"
OUTPUT="${OUTPUT:-results_walker_actuator.csv}"
EPISODES="${EPISODES:-30}"
MODEL_SIZE="${MODEL_SIZE:-5}"
LOGS_DIR="${LOGS_DIR:-logs}"

echo "Evaluating Walker actuator models..."
echo "  Seeds: ${SEEDS}"
echo "  Episodes: ${EPISODES}"
echo "  Logs dir: ${LOGS_DIR}"
echo "  Output: ${OUTPUT}"

python evaluate/evaluate_walker_actuator_all_models.py \
    --seeds ${SEEDS} \
    --episodes ${EPISODES} \
    --actuator-scales 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 \
    --output ${OUTPUT} \
    --logs-dir ${LOGS_DIR} \
    --model-size ${MODEL_SIZE}

echo "Walker actuator evaluation completed."
