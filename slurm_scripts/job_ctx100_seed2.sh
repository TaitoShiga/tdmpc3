#!/bin/bash
#SBATCH -J tdmpc2-ctx100-seed2
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

eval "$(conda shell.bash hook)"
conda activate tdmpc2

mkdir -p logs
cd ~/tdmpc3/tdmpc3

python tdmpc2/train.py task=pendulum-swingup-randomized use_model_c=true \
    context_length=100 seed=2 steps=100000 \
    exp_name=modelc_ctx100 log_interval=100 eval_freq=500 compile=false enable_wandb=false save_video=false

