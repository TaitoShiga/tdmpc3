#!/bin/bash
# Context Length 10 - 全seeds並列実行

echo "=========================================="
echo "Context Length 10 (全seeds並列)"
echo "=========================================="

sbatch slurm_scripts/job_ctx10_seed0.sh
echo "Submitted: ctx=10, seed=0"
sleep 0.5

sbatch slurm_scripts/job_ctx10_seed1.sh
echo "Submitted: ctx=10, seed=1"
sleep 0.5

sbatch slurm_scripts/job_ctx10_seed2.sh
echo "Submitted: ctx=10, seed=2"
sleep 0.5

echo "=========================================="
echo "Context Length 10: 全3ジョブ投入完了"
echo "=========================================="

