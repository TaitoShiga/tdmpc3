#!/bin/bash
#SBATCH -J walker-act-dr-seed4
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

eval "$(conda shell.bash hook)"
conda activate tdmpc2
mkdir -p logs
cd ~/tdmpc3/tdmpc3

echo "Training Actuator DR seed=4..."
python tdmpc2/train.py \
    task=walker-walk_actuator_randomized \
    exp_name=walker_actuator_dr \
    steps=100000 \
    eval_freq=500 \
    seed=4 \
    save_video=false \
    enable_wandb=false \
    compile=false

echo "Actuator DR seed=4 completed!"
