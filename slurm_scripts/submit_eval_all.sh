#!/bin/bash
# 評価ジョブ一括投入スクリプト: 5ジョブ（seed 0-4）を投入
# 使用方法: bash submit_eval_all.sh

echo "Starting batch submission of 5 evaluation jobs..."
echo "=============================================="
echo ""
echo "Each job evaluates:"
echo "  - 4 models: baseline, dr, Model C, Oracle"
echo "  - 5 params: 0.5, 1.0, 1.5, 2.0, 2.5"
echo "  - 30 episodes per (model, param)"
echo ""
echo "Total per job: 4 × 5 × 30 = 600 episodes"
echo "Estimated time per job: ~2-4 hours"
echo ""
echo "=============================================="

# Evaluation jobs (seed 0-4)
echo "Submitting evaluation jobs..."
for seed in 0 1 2 3 4; do
    sbatch slurm_scripts/job_eval_seed${seed}.sh
    echo "  ✓ Submitted: job_eval_seed${seed}.sh"
    sleep 0.5  # 少し間隔を空ける
done

echo ""
echo "=============================================="
echo "All 5 evaluation jobs submitted successfully!"
echo ""
echo "Check job status with:"
echo "  squeue -u \$USER"
echo ""
echo "Monitor specific job:"
echo "  tail -f logs/tdmpc2-eval-seed<N>-<jobid>.out"
echo ""
echo "After all jobs complete, merge results:"
echo "  cat results_seed*.csv | head -1 > results.csv"
echo "  for i in 0 1 2 3 4; do tail -n +2 results_seed\$i.csv >> results.csv; done"
echo ""
echo "Then run statistical analysis:"
echo "  python evaluate/analyze_results.py"

