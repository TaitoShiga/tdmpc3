#!/bin/bash
# Context Length 50 - 全seeds並列実行

echo "=========================================="
echo "Context Length 50 (全seeds並列)"
echo "=========================================="

sbatch slurm_scripts/job_ctx50_seed0.sh
echo "Submitted: ctx=50, seed=0"
sleep 0.5

sbatch slurm_scripts/job_ctx50_seed1.sh
echo "Submitted: ctx=50, seed=1"
sleep 0.5

sbatch slurm_scripts/job_ctx50_seed2.sh
echo "Submitted: ctx=50, seed=2"
sleep 0.5

echo "=========================================="
echo "Context Length 50: 全3ジョブ投入完了"
echo "=========================================="

