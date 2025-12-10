#!/bin/bash
# Context Length 200 - 全seeds並列実行

echo "=========================================="
echo "Context Length 200 (全seeds並列)"
echo "=========================================="

sbatch slurm_scripts/job_ctx200_seed0.sh
echo "Submitted: ctx=200, seed=0"
sleep 0.5

sbatch slurm_scripts/job_ctx200_seed1.sh
echo "Submitted: ctx=200, seed=1"
sleep 0.5

sbatch slurm_scripts/job_ctx200_seed2.sh
echo "Submitted: ctx=200, seed=2"
sleep 0.5

echo "=========================================="
echo "Context Length 200: 全3ジョブ投入完了"
echo "=========================================="

