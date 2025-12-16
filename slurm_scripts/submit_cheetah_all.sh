#!/bin/bash
# Cheetah-Run 全モデル・全Seeds一括投入スクリプト
# Usage: bash submit_cheetah_all.sh

echo "========================================"
echo "Cheetah-Run Slurm Jobs Submission"
echo "========================================"
echo ""
echo "Models: Baseline, DR, Model C, Oracle"
echo "Seeds: 0, 1, 2, 3, 4"
echo "Total: 20 jobs (4 models × 5 seeds)"
echo ""

# Baseline (seeds 0-4)
echo "Submitting Baseline jobs..."
sbatch slurm_scripts/job_cheetah_baseline_seed0.sh
sbatch slurm_scripts/job_cheetah_baseline_seed1.sh
sbatch slurm_scripts/job_cheetah_baseline_seed2.sh
sbatch slurm_scripts/job_cheetah_baseline_seed3.sh
sbatch slurm_scripts/job_cheetah_baseline_seed4.sh

# DR (seeds 0-4)
echo "Submitting DR jobs..."
sbatch slurm_scripts/job_cheetah_dr_seed0.sh
sbatch slurm_scripts/job_cheetah_dr_seed1.sh
sbatch slurm_scripts/job_cheetah_dr_seed2.sh
sbatch slurm_scripts/job_cheetah_dr_seed3.sh
sbatch slurm_scripts/job_cheetah_dr_seed4.sh

# Model C (seeds 0-4)
echo "Submitting Model C jobs..."
sbatch slurm_scripts/job_cheetah_modelc_seed0.sh
sbatch slurm_scripts/job_cheetah_modelc_seed1.sh
sbatch slurm_scripts/job_cheetah_modelc_seed2.sh
sbatch slurm_scripts/job_cheetah_modelc_seed3.sh
sbatch slurm_scripts/job_cheetah_modelc_seed4.sh

# Oracle (seeds 0-4)
echo "Submitting Oracle jobs..."
sbatch slurm_scripts/job_cheetah_oracle_seed0.sh
sbatch slurm_scripts/job_cheetah_oracle_seed1.sh
sbatch slurm_scripts/job_cheetah_oracle_seed2.sh
sbatch slurm_scripts/job_cheetah_oracle_seed3.sh
sbatch slurm_scripts/job_cheetah_oracle_seed4.sh

echo ""
echo "========================================"
echo "✅ All 20 jobs submitted!"
echo "========================================"
echo ""
echo "Check job status:"
echo "  squeue -u \$USER"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/cheetah-*.out"
echo ""
echo "Expected completion: 24-48 hours"
echo "========================================"

