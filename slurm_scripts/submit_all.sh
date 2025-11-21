#!/bin/bash
# 一括投入スクリプト: 20ジョブすべてを投入
# 使用方法: bash submit_all.sh

echo "Starting batch submission of 20 tdmpc2 jobs..."
echo "=============================================="

# Baseline (seed 0-4)
echo "Submitting Baseline jobs..."
for seed in 0 1 2 3 4; do
    sbatch job_baseline_seed${seed}.sh
    echo "  - Submitted: job_baseline_seed${seed}.sh"
done

# DR (Domain Randomization, seed 0-4)
echo "Submitting DR jobs..."
for seed in 0 1 2 3 4; do
    sbatch job_dr_seed${seed}.sh
    echo "  - Submitted: job_dr_seed${seed}.sh"
done

# Model O (Oracle, seed 0-4)
echo "Submitting Oracle jobs..."
for seed in 0 1 2 3 4; do
    sbatch job_oracle_seed${seed}.sh
    echo "  - Submitted: job_oracle_seed${seed}.sh"
done

# Model C (Proposed Method, seed 0-4)
echo "Submitting Model C jobs..."
for seed in 0 1 2 3 4; do
    sbatch job_modelc_seed${seed}.sh
    echo "  - Submitted: job_modelc_seed${seed}.sh"
done

echo "=============================================="
echo "All 20 jobs submitted successfully!"
echo ""
echo "Check job status with:"
echo "  squeue -u shiga-t"
echo ""
echo "Monitor specific job:"
echo "  tail -f logs/tdmpc2-<model>-seed<N>-<jobid>.out"

