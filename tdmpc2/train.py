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
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

# Oracle imports (conditional)
try:
	from common.buffer_oracle import OracleBuffer
	from tdmpc2_oracle import TDMPC2Oracle
	from trainer.online_trainer_oracle import OnlineTrainerOracle
	from envs.wrappers.physics_param import wrap_with_physics_param
	ORACLE_AVAILABLE = True
except ImportError:
	ORACLE_AVAILABLE = False

# Model C imports (conditional)
try:
	from common.buffer_model_c import ModelCBuffer
	from tdmpc2_model_c import TDMPC2ModelC
	from trainer.online_trainer_model_c import OnlineTrainerModelC
	MODEL_C_AVAILABLE = True
except ImportError:
	MODEL_C_AVAILABLE = False

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	
	# Mode selection
	use_oracle = getattr(cfg, 'use_oracle', False)
	use_model_c = getattr(cfg, 'use_model_c', False)
	
	# Validation
	if use_oracle and use_model_c:
		raise ValueError('Cannot use both Oracle and Model C modes simultaneously.')
	
	if use_oracle and not ORACLE_AVAILABLE:
		raise ImportError(
			'Oracle mode is enabled but Oracle components are not available. '
			'Make sure all Oracle files are present in the tdmpc2 directory.'
		)
	
	if use_model_c and not MODEL_C_AVAILABLE:
		raise ImportError(
			'Model C mode is enabled but Model C components are not available. '
			'Make sure all Model C files are present in the tdmpc2 directory.'
		)
	
	# Print mode
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
	if use_model_c:
		print(colored('Mode:', 'cyan', attrs=['bold']), 'Model C (GRU physics estimator)')
		print(colored('c_phys_dim:', 'cyan'), cfg.c_phys_dim)
		print(colored('context_length:', 'cyan'), cfg.context_length)
		print(colored('gru_hidden_dim:', 'cyan'), cfg.gru_hidden_dim)
	elif use_oracle:
		print(colored('Mode:', 'cyan', attrs=['bold']), 'Oracle (using true physics parameters)')
		print(colored('c_phys_dim:', 'cyan'), cfg.c_phys_dim)
	else:
		print(colored('Mode:', 'cyan', attrs=['bold']), 'Standard')
	
	# Create environment
	env = make_env(cfg)
	if use_oracle or use_model_c:
		env = wrap_with_physics_param(env, cfg)
		print(colored('Physics parameter wrapper enabled:', 'green'))
		print(colored(f'  - param_type: {env.param_type}', 'green'))
		print(colored(f'  - normalization: {env.normalization}', 'green'))
		print(colored(f'  - default_value: {env.default_value}', 'green'))
	
	# Select components based on mode
	if cfg.multitask:
		trainer_cls = OfflineTrainer
		agent_cls = TDMPC2
		buffer_cls = Buffer
	elif use_model_c:
		trainer_cls = OnlineTrainerModelC
		agent_cls = TDMPC2ModelC
		buffer_cls = ModelCBuffer
	elif use_oracle:
		trainer_cls = OnlineTrainerOracle
		agent_cls = TDMPC2Oracle
		buffer_cls = OracleBuffer
	else:
		trainer_cls = OnlineTrainer
		agent_cls = TDMPC2
		buffer_cls = Buffer
	
	# Create agent
	agent = agent_cls(cfg)
	
	# Load pretrained GRU (Model C only)
	if use_model_c and hasattr(cfg, 'gru_pretrained') and cfg.gru_pretrained:
		# Check if gru_pretrained is a valid path (not a placeholder like '???')
		if cfg.gru_pretrained not in ['???', 'null', '', None]:
			print(colored(f'Loading pretrained GRU from: {cfg.gru_pretrained}', 'magenta'))
			agent.load_pretrained_gru(cfg.gru_pretrained)
		else:
			print(colored('Training GRU from scratch (no pretrained model specified)', 'yellow'))
	
	# Create trainer
	trainer = trainer_cls(
		cfg=cfg,
		env=env,
		agent=agent,
		buffer=buffer_cls(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
