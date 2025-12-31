#!/bin/bash
#SBATCH -J walker-eval-oracle
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

echo "Evaluating Walker Oracle (seed0) on 8 mass conditions..."

CHECKPOINT="$PWD/logs/walker-walk_randomized/0/walker_oracle/models/final.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo ""

# Oracle用の物理パラメータ設定
PHYS_PARAMS="use_oracle=true c_phys_dim=1 phys_param_type=mass phys_param_normalization=standard"

# In-Distribution
for TASK in walk_torso_mass_05x walk_torso_mass_10x walk_torso_mass_15x walk_torso_mass_20x walk_torso_mass_25x; do
    echo "Evaluating $TASK..."
    python tdmpc2/evaluate.py \
        task=walker-${TASK} \
        checkpoint=${CHECKPOINT} \
        eval_episodes=30 \
        seed=0 \
        save_video=false \
        compile=false \
        ${PHYS_PARAMS}
done

# Out-of-Distribution
for TASK in walk_torso_mass_03x walk_torso_mass_30x walk_torso_mass_35x; do
    echo "Evaluating $TASK (OOD)..."
    python tdmpc2/evaluate.py \
        task=walker-${TASK} \
        checkpoint=${CHECKPOINT} \
        eval_episodes=30 \
        seed=0 \
        save_video=false \
        compile=false \
        ${PHYS_PARAMS}
done

echo "Oracle evaluation completed!"
