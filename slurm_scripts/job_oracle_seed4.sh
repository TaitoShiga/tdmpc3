#!/bin/bash
#SBATCH -J tdmpc2-oracle-seed4
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# モジュールと conda の初期化
source /etc/profile.d/modules.sh
module load slurm/23.02.7
source ~/.bashrc
conda activate tdmpc2

# ログディレクトリ作成
mkdir -p logs

# 作業ディレクトリへ移動
cd ~/tdmpc3/tdmpc3

# 実験コマンド: Model O (Oracle), Seed 4
python tdmpc2/train.py task=pendulum-swingup-randomized use_oracle=true seed=4 steps=100000 \
    exp_name=oracle log_interval=100 compile=false enable_wandb=false

