#!/bin/bash
#SBATCH -J walker-dr-seed2
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

echo "Training DR seed=2..."
python tdmpc2/train.py \
    task=walker-walk_randomized \
    exp_name=walker_dr \
    steps=100000 \
    eval_freq=500 \
    seed=2 \
    save_video=false \
    enable_wandb=false \
    compile=false

echo "DR seed=2 completed!"

