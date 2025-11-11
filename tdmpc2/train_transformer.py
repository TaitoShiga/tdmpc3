"""Training script for Transformer-based TD-MPC2

Domain Randomization環境でTransformerモデルを訓練。
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
from common.trajectory_buffer import TrajectoryBuffer
from envs import make_env
from tdmpc2_transformer import TDMPC2Transformer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""Train Transformer-based TD-MPC2 agent
	
	Most relevant args:
		task: pendulum-swingup-randomized (Domain Randomization)
		model_size: model size (default: 5)
		context_length: Transformer context length (default: 50)
		transformer_layers: number of Transformer layers (default: 4)
		transformer_heads: number of attention heads (default: 8)
		steps: training steps (default: 500K)
		seed: random seed
	
	Example usage:
	```
		$ python train_transformer.py task=pendulum-swingup-randomized seed=0
		$ python train_transformer.py context_length=100 transformer_layers=6
	```
	"""
	assert torch.cuda.is_available(), "CUDA is required for training"
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	
	print(colored('=' * 70, 'cyan', attrs=['bold']))
	print(colored('Transformer-based TD-MPC2 Training', 'cyan', attrs=['bold']))
	print(colored('=' * 70, 'cyan', attrs=['bold']))
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
	print(colored('Task:', 'yellow', attrs=['bold']), cfg.task)
	print(colored('Context length:', 'yellow', attrs=['bold']), 
		  getattr(cfg, 'context_length', 50))
	print(colored('Transformer layers:', 'yellow', attrs=['bold']), 
		  getattr(cfg, 'transformer_layers', 4))
	print(colored('Attention heads:', 'yellow', attrs=['bold']), 
		  getattr(cfg, 'transformer_heads', 8))
	print(colored('Steps:', 'yellow', attrs=['bold']), cfg.steps)
	print(colored('=' * 70, 'cyan', attrs=['bold']))
	
	# Transformerモデル用の設定確認
	if not hasattr(cfg, 'context_length'):
		cfg.context_length = 50
		print(colored('Warning:', 'yellow'), 
			  f'context_length not set, using default: {cfg.context_length}')
	
	if not hasattr(cfg, 'transformer_layers'):
		cfg.transformer_layers = 4
		print(colored('Warning:', 'yellow'), 
			  f'transformer_layers not set, using default: {cfg.transformer_layers}')
	
	if not hasattr(cfg, 'transformer_heads'):
		cfg.transformer_heads = 8
		print(colored('Warning:', 'yellow'), 
			  f'transformer_heads not set, using default: {cfg.transformer_heads}')
	
	# Transformerエージェントを作成
	trainer = OnlineTrainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2Transformer(cfg),
		buffer=TrajectoryBuffer(cfg),
		logger=Logger(cfg),
	)
	
	trainer.train()
	print(colored('\n✅ Transformer training completed successfully', 'green', attrs=['bold']))


if __name__ == '__main__':
	train()

