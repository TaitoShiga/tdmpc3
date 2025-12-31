#!/bin/bash
# Walker Walk Zero-shot評価を4モデル並列で投入

echo "=========================================="
echo "Walker Walk Zero-shot Evaluation"
echo "=========================================="
echo "Submitting 4 evaluation jobs (parallel)..."
echo ""

# 各モデルを並列で評価
sbatch slurm_scripts/evaluate_walker_baseline.sh
sbatch slurm_scripts/evaluate_walker_dr.sh
sbatch slurm_scripts/evaluate_walker_modelc.sh
sbatch slurm_scripts/evaluate_walker_oracle.sh

echo ""
echo "All jobs submitted!"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: logs/walker-eval-*.out"

