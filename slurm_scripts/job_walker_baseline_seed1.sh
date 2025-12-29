#!/bin/bash
#SBATCH -J walker-baseline-seed1
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

echo "Training Baseline seed=1..."
python tdmpc2/train.py \
    task=walker-walk \
    exp_name=walker_baseline \
    steps=100000 \
    eval_freq=500 \
    seed=1 \
    save_video=false \
    enable_wandb=false \
    compile=false

echo "Baseline seed=1 completed!"

