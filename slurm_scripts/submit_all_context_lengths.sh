#!/bin/bash
# Context Length Ablation - 全context lengthを並列実行

echo "=========================================="
echo "Context Length Ablation (並列実行)"
echo "=========================================="
echo "Context Lengths: 10, 25, 50, 100, 200"
echo "Seed: 0"
echo "=========================================="

# 各context lengthのジョブを投入（並列実行）
sbatch slurm_scripts/job_ctx10_seed0.sh
echo "Submitted: Context Length 10"
sleep 1

sbatch slurm_scripts/job_ctx25_seed0.sh
echo "Submitted: Context Length 25"
sleep 1

sbatch slurm_scripts/job_ctx50_seed0.sh
echo "Submitted: Context Length 50"
sleep 1

sbatch slurm_scripts/job_ctx100_seed0.sh
echo "Submitted: Context Length 100"
sleep 1

sbatch slurm_scripts/job_ctx200_seed0.sh
echo "Submitted: Context Length 200"
sleep 1

echo "=========================================="
echo "全5ジョブ投入完了！"
echo "=========================================="
echo ""
echo "ジョブ確認: squeue -u \$USER"
echo "ログ確認: tail -f logs/tdmpc2-ctx*-seed0-*.out"
echo ""
echo "各ジョブは並列実行されます（GPU 5台必要）"
echo "所要時間: 約12-18時間"

