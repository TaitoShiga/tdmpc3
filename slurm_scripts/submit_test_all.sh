#!/bin/bash
# テスト用一括投入スクリプト: 4モデルすべてをテスト（各5000ステップ）
# 使用方法: bash submit_test_all.sh

echo "Starting test batch submission (5000 steps each)..."
echo "=============================================="

echo "Submitting test jobs..."
sbatch job_test_baseline.sh
echo "  - Submitted: job_test_baseline.sh"

sbatch job_test_dr.sh
echo "  - Submitted: job_test_dr.sh"

sbatch job_test_oracle.sh
echo "  - Submitted: job_test_oracle.sh"

sbatch job_test_modelc.sh
echo "  - Submitted: job_test_modelc.sh"

echo "=============================================="
echo "All 4 test jobs submitted!"
echo ""
echo "Check job status with:"
echo "  squeue -u shiga-t"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/tdmpc2-test-baseline-*.out"
echo "  tail -f logs/tdmpc2-test-dr-*.out"
echo "  tail -f logs/tdmpc2-test-oracle-*.out"
echo "  tail -f logs/tdmpc2-test-modelc-*.out"

