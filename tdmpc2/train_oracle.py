"""
Oracle版学習スクリプト

Model O (Oracle) の学習用エントリーポイント。

真の物理パラメータを常にプランナーに注入することで、
「物理推定が完璧な場合の理論的上限」を検証する。

使用例:
    python train_oracle.py task=pendulum-swingup seed=0
    python train_oracle.py task=pendulum-swingup seed=1
    python train_oracle.py task=ball_in_cup-catch seed=0
"""
import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer_oracle import OracleBuffer
from envs import make_env
from envs.wrappers.physics_param import wrap_with_physics_param
from tdmpc2_oracle import TDMPC2Oracle
from trainer.online_trainer_oracle import OnlineTrainerOracle
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config_oracle', config_path='.')
def train(cfg: dict):
	"""
	Oracle版TD-MPC2エージェントの学習スクリプト。
	
	主な引数:
		`task`: タスク名（例: pendulum-swingup, ball_in_cup-catch）
		`model_size`: モデルサイズ [1, 5, 19, 48, 317] (デフォルト: 5)
		`steps`: 学習ステップ数（デフォルト: 500,000）
		`seed`: ランダムシード（デフォルト: 0）
		`c_phys_dim`: 物理パラメータの次元（デフォルト: 1）
	
	使用例:
		$ python train_oracle.py task=pendulum-swingup seed=0
		$ python train_oracle.py task=ball_in_cup-catch seed=1 steps=1000000
	"""
	assert torch.cuda.is_available(), 'CUDA is required for training.'
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	
	print(colored('='*70, 'yellow', attrs=['bold']))
	print(colored('Model O (Oracle) Training', 'yellow', attrs=['bold']))
	print(colored('='*70, 'yellow', attrs=['bold']))
	print(colored('Task:', 'cyan', attrs=['bold']), cfg.task)
	print(colored('c_phys_dim:', 'cyan', attrs=['bold']), cfg.c_phys_dim)
	print(colored('Work dir:', 'cyan', attrs=['bold']), cfg.work_dir)
	print(colored('='*70, 'yellow', attrs=['bold']))
	
	# 環境の作成
	env = make_env(cfg)
	
	# 物理パラメータ取得Wrapperを追加
	env = wrap_with_physics_param(env, cfg)
	print(colored('Physics parameter wrapper enabled.', 'green'))
	print(colored(f'  - c_phys_dim: {env.c_phys_dim}', 'green'))
	print(colored(f'  - normalization: {env.normalization}', 'green'))
	print(colored(f'  - default_value: {env.default_value}', 'green'))
	print(colored(f'  - scale: {env.scale}', 'green'))
	
	# Oracleエージェント、バッファ、ロガーの作成
	agent = TDMPC2Oracle(cfg)
	buffer = OracleBuffer(cfg)
	logger = Logger(cfg)
	
	print(colored('\nModel architecture:', 'magenta', attrs=['bold']))
	print(agent.model)
	
	# Trainerの作成と学習開始
	trainer = OnlineTrainerOracle(
		cfg=cfg,
		env=env,
		agent=agent,
		buffer=buffer,
		logger=logger,
	)
	
	print(colored('\nStarting training...', 'green', attrs=['bold']))
	trainer.train()
	print(colored('\nTraining completed successfully!', 'green', attrs=['bold']))


if __name__ == '__main__':
	train()

