#!/bin/bash
#SBATCH -J walker-eval-seed0
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

eval "$(conda shell.bash hook)"
conda activate tdmpc2
mkdir -p logs
cd ~/tdmpc3/tdmpc3

echo "=========================================="
echo "Walker Walk Zero-shot Evaluation (seed0)"
echo "=========================================="
echo "Start time: $(date)"
echo ""
echo "Models: Baseline, DR, Model C, Oracle"
echo "Tasks: 8 mass conditions (5 In-Dist + 3 OOD)"
echo "Episodes per task: 30"
echo ""

python scripts/evaluate_walker_all_models.py

echo ""
echo "Evaluation completed!"
echo "End time: $(date)"

