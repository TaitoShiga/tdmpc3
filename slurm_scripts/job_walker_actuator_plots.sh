#!/bin/bash
#SBATCH -J walker-actuator-plots
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

eval "$(conda shell.bash hook)"
conda activate tdmpc2
mkdir -p logs
cd ~/tdmpc3/tdmpc3

RESULTS="${RESULTS:-results_walker_actuator.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"
LOGS_DIR="${LOGS_DIR:-logs}"

if [ ! -f "${RESULTS}" ]; then
    echo "Error: results CSV not found: ${RESULTS}"
    exit 1
fi

echo "Generating evaluation plots..."
python evaluate/analyze_results.py \
    --input ${RESULTS} \
    --output-dir ${OUTPUT_DIR} \
    --output-prefix walker_actuator \
    --param-label "Actuator scale (x)"

echo "Generating learning curves..."
python plot_learning_curves.py \
    --task walker_actuator \
    --logs-dir ${LOGS_DIR} \
    --output eval_curves_walker_actuator.png

echo "Walker actuator plots completed."
