#!/bin/bash
#SBATCH -J walker-eval-dr
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

eval "$(conda shell.bash hook)"
conda activate tdmpc2
mkdir -p logs
cd ~/tdmpc3/tdmpc3

echo "Evaluating Walker DR (seed0) on 8 mass conditions..."

CHECKPOINT=$(find logs/walker_dr/seed0* -name "checkpoint.pt" | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found for walker_dr/seed0"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo ""

# In-Distribution
for TASK in walk_torso_mass_05x walk_torso_mass_10x walk_torso_mass_15x walk_torso_mass_20x walk_torso_mass_25x; do
    echo "Evaluating $TASK..."
    python tdmpc2/evaluate.py \
        task=walker-${TASK} \
        checkpoint=${CHECKPOINT} \
        episodes=30 \
        seed=0 \
        save_video=false
done

# Out-of-Distribution
for TASK in walk_torso_mass_03x walk_torso_mass_30x walk_torso_mass_35x; do
    echo "Evaluating $TASK (OOD)..."
    python tdmpc2/evaluate.py \
        task=walker-${TASK} \
        checkpoint=${CHECKPOINT} \
        episodes=30 \
        seed=0 \
        save_video=false
done

echo "DR evaluation completed!"

