#!/bin/bash
# Context Length Ablation - 全context length × 3 seedsを並列実行

echo "=========================================="
echo "Context Length Ablation (並列実行)"
echo "=========================================="
echo "Context Lengths: 10, 25, 50, 100, 200"
echo "Seeds: 0, 1, 2"
echo "Total jobs: 15 (5 context lengths × 3 seeds)"
echo "=========================================="

# Seed 0
echo "Seed 0 ..."
sbatch slurm_scripts/job_ctx10_seed0.sh
sbatch slurm_scripts/job_ctx25_seed0.sh
sbatch slurm_scripts/job_ctx50_seed0.sh
sbatch slurm_scripts/job_ctx100_seed0.sh
sbatch slurm_scripts/job_ctx200_seed0.sh
sleep 1

# Seed 1
echo "Seed 1 ..."
sbatch slurm_scripts/job_ctx10_seed1.sh
sbatch slurm_scripts/job_ctx25_seed1.sh
sbatch slurm_scripts/job_ctx50_seed1.sh
sbatch slurm_scripts/job_ctx100_seed1.sh
sbatch slurm_scripts/job_ctx200_seed1.sh
sleep 1

# Seed 2
echo "Seed 2 ..."
sbatch slurm_scripts/job_ctx10_seed2.sh
sbatch slurm_scripts/job_ctx25_seed2.sh
sbatch slurm_scripts/job_ctx50_seed2.sh
sbatch slurm_scripts/job_ctx100_seed2.sh
sbatch slurm_scripts/job_ctx200_seed2.sh
sleep 1

echo "=========================================="
echo "全15ジョブ投入完了！"
echo "=========================================="
echo ""
echo "ジョブ確認: squeue -u \$USER"
echo "ログ確認: tail -f logs/tdmpc2-ctx*-*.out"
echo ""
echo "各ジョブは並列実行されます（GPU 15台必要）"
echo "所要時間: 約12-18時間"

