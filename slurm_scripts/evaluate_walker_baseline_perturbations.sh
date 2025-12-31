#!/bin/bash
#SBATCH -J walker-eval-perturb
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

eval "$(conda shell.bash hook)"
conda activate tdmpc2
mkdir -p logs results
cd ~/tdmpc3/tdmpc3

echo "Evaluating Walker Baseline across perturbations..."

CHECKPOINT="$PWD/logs/walker-walk/0/walker_baseline/models/final.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo ""

python scripts/evaluate_walker_baseline_perturbations.py \
    --checkpoint "$CHECKPOINT" \
    --eval-episodes 30 \
    --seed 0 \
    --output results/walker_baseline_perturbations.csv

echo "Baseline perturbation evaluation completed!"
echo "Results saved to: results/walker_baseline_perturbations.csv"
