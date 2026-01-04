#!/bin/bash
# Submit Walker actuator evaluation jobs per seed.

echo "Submitting Walker actuator evaluation jobs..."
for seed in 0 1 2 3 4; do
    sbatch \
        --job-name=walker-act-eval-seed${seed} \
        --export=SEEDS=${seed},OUTPUT=results_walker_actuator_seed${seed}.csv,LOGS_DIR=logs \
        slurm_scripts/job_walker_actuator_eval.sh
    echo "  - Submitted seed ${seed}"
    sleep 0.5
done

echo ""
echo "After jobs complete, merge results:"
echo "  head -1 results_walker_actuator_seed0.csv > results_walker_actuator.csv"
echo "  for i in 0 1 2 3 4; do tail -n +2 results_walker_actuator_seed\$i.csv >> results_walker_actuator.csv; done"
echo ""
echo "Then generate plots:"
echo "  sbatch slurm_scripts/job_walker_actuator_plots.sh"
