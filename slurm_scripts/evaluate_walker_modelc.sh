#!/bin/bash
#SBATCH -J walker-eval-modelc
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

echo "Evaluating Walker Model C (seed0) on 8 mass conditions..."

CHECKPOINT="logs/walker-walk_randomized/0/walker_model_c/models/final.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo ""

# Model C用の物理パラメータ設定
PHYS_PARAMS="use_model_c=true c_phys_dim=1 phys_param_type=mass phys_param_normalization=standard context_length=50 gru_hidden_dim=256"

# In-Distribution
for TASK in walk_torso_mass_05x walk_torso_mass_10x walk_torso_mass_15x walk_torso_mass_20x walk_torso_mass_25x; do
    echo "Evaluating $TASK..."
    python tdmpc2/evaluate.py \
        task=walker-${TASK} \
        checkpoint=${CHECKPOINT} \
        eval_episodes=30 \
        seed=0 \
        save_video=false \
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
        ${PHYS_PARAMS}
done

echo "Model C evaluation completed!"

