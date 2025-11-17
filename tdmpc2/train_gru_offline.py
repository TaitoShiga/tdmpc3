"""
GRU推定器のオフライン学習スクリプト

Model Bのリプレイバッファから収集したデータを使って、
GRU推定器を事前学習する。

使用方法:
    python train_gru_offline.py \
        task=pendulum-swingup-randomized \
        buffer_path=logs/pendulum-swingup-randomized/0/buffer.pkl \
        context_length=50 \
        estimator_type=gru
"""
import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.physics_estimator import create_physics_estimator
from envs import make_env
from envs.wrappers.physics_param import wrap_with_physics_param


class GRUOfflineTrainer:
	"""GRU推定器のオフライン学習"""
	
	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		
		# 環境（物理パラメータ抽出用）
		env = make_env(cfg)
		self.env = wrap_with_physics_param(env, cfg)
		
		# 推定器の作成
		self.estimator = create_physics_estimator(
			estimator_type=cfg.estimator_type,
			obs_dim=self.env.observation_space.shape[0],
			action_dim=self.env.action_space.shape[0],
			c_phys_dim=cfg.c_phys_dim,
			hidden_dim=cfg.gru_hidden_dim,
			num_layers=cfg.gru_num_layers,
			dropout=cfg.gru_dropout,
			context_length=cfg.context_length,
		).to(self.device)
		
		# Optimizer
		self.optimizer = torch.optim.Adam(
			self.estimator.parameters(),
			lr=cfg.gru_lr,
			weight_decay=cfg.gru_weight_decay,
		)
		
		# Scheduler（オプション）
		self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			self.optimizer,
			T_max=cfg.gru_epochs,
			eta_min=cfg.gru_lr * 0.1,
		)
		
		print(colored('\n=== GRU Offline Trainer ===', 'cyan', attrs=['bold']))
		print(colored(f'Estimator type: {cfg.estimator_type}', 'white'))
		print(colored(f'Context length: {cfg.context_length}', 'white'))
		print(colored(f'Hidden dim: {cfg.gru_hidden_dim}', 'white'))
		print(colored(f'Num layers: {cfg.gru_num_layers}', 'white'))
		print(colored(f'Total params: {sum(p.numel() for p in self.estimator.parameters()):,}', 'white'))
	
	def collect_data_from_oracle_buffer(self, num_episodes=1000):
		"""
		Oracle版の学習からデータを収集。
		
		実際には、環境を動かしてDRでデータを収集。
		"""
		print(colored('\nCollecting data from environment...', 'cyan'))
		
		data = {
			'obs_seq': [],
			'action_seq': [],
			'c_phys': [],
		}
		
		for ep in tqdm(range(num_episodes), desc='Collecting episodes'):
			obs = self.env.reset()
			c_phys = self.env.current_c_phys
			
			episode_obs = []
			episode_actions = []
			
			done = False
			t = 0
			while not done and t < self.cfg.episode_length:
				# ランダム方策
				action = self.env.rand_act()
				
				episode_obs.append(obs.numpy())
				episode_actions.append(action.numpy())
				
				obs, reward, done, info = self.env.step(action)
				t += 1
			
			# 十分な長さがあればデータとして保存
			if len(episode_obs) >= self.cfg.context_length:
				data['obs_seq'].append(np.array(episode_obs))
				data['action_seq'].append(np.array(episode_actions))
				data['c_phys'].append(c_phys.numpy())
		
		# Tensorに変換
		self.data = {
			'obs_seq': data['obs_seq'],
			'action_seq': data['action_seq'],
			'c_phys': np.array(data['c_phys']),
		}
		
		print(colored(f'✓ Collected {len(data["obs_seq"])} episodes', 'green'))
		print(colored(f'  c_phys range: [{self.data["c_phys"].min():.3f}, {self.data["c_phys"].max():.3f}]', 'white'))
		
		return self.data
	
	def create_dataloader(self, split='train', batch_size=32):
		"""データローダーを作成"""
		num_episodes = len(self.data['obs_seq'])
		num_train = int(num_episodes * 0.8)
		
		if split == 'train':
			indices = list(range(num_train))
		else:
			indices = list(range(num_train, num_episodes))
		
		# データセットの作成
		dataset = []
		for idx in indices:
			obs_seq = self.data['obs_seq'][idx]
			action_seq = self.data['action_seq'][idx]
			c_phys = self.data['c_phys'][idx]
			
			# context_length分のウィンドウを切り出し
			max_start = len(obs_seq) - self.cfg.context_length
			if max_start > 0:
				# ランダムに開始位置を選択
				for _ in range(self.cfg.num_samples_per_episode):
					start = np.random.randint(0, max_start)
					end = start + self.cfg.context_length
					
					dataset.append({
						'obs_seq': torch.from_numpy(obs_seq[start:end]).float(),
						'action_seq': torch.from_numpy(action_seq[start:end]).float(),
						'c_phys': torch.from_numpy(c_phys).float(),
					})
		
		# DataLoader
		dataloader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=(split == 'train'),
			num_workers=0,
			pin_memory=True,
		)
		
		return dataloader
	
	def train_epoch(self, train_loader):
		"""1エポックの学習"""
		self.estimator.train()
		
		total_loss = 0.0
		total_mae = 0.0
		num_batches = 0
		
		for batch in train_loader:
			obs_seq = batch['obs_seq'].to(self.device)
			action_seq = batch['action_seq'].to(self.device)
			c_phys_true = batch['c_phys'].to(self.device)
			
			# Forward
			loss, info = self.estimator.compute_loss(obs_seq, action_seq, c_phys_true)
			
			# Backward
			self.optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), self.cfg.gru_grad_clip)
			self.optimizer.step()
			
			total_loss += info['loss']
			total_mae += info['mae']
			num_batches += 1
		
		return {
			'loss': total_loss / num_batches,
			'mae': total_mae / num_batches,
		}
	
	def evaluate(self, val_loader):
		"""検証"""
		self.estimator.eval()
		
		total_loss = 0.0
		total_mae = 0.0
		num_batches = 0
		
		predictions = []
		targets = []
		
		with torch.no_grad():
			for batch in val_loader:
				obs_seq = batch['obs_seq'].to(self.device)
				action_seq = batch['action_seq'].to(self.device)
				c_phys_true = batch['c_phys'].to(self.device)
				
				loss, info = self.estimator.compute_loss(obs_seq, action_seq, c_phys_true)
				
				total_loss += info['loss']
				total_mae += info['mae']
				num_batches += 1
				
				# 予測値を保存（可視化用）
				c_phys_pred, _ = self.estimator(obs_seq, action_seq)
				predictions.append(c_phys_pred.cpu().numpy())
				targets.append(c_phys_true.cpu().numpy())
		
		predictions = np.concatenate(predictions, axis=0)
		targets = np.concatenate(targets, axis=0)
		
		return {
			'loss': total_loss / num_batches,
			'mae': total_mae / num_batches,
			'predictions': predictions,
			'targets': targets,
		}
	
	def train(self):
		"""メインの学習ループ"""
		# データ収集
		self.collect_data_from_oracle_buffer(num_episodes=self.cfg.num_episodes)
		
		# DataLoader
		train_loader = self.create_dataloader('train', batch_size=self.cfg.gru_batch_size)
		val_loader = self.create_dataloader('val', batch_size=self.cfg.gru_batch_size)
		
		print(colored(f'\nTrain samples: {len(train_loader.dataset)}', 'white'))
		print(colored(f'Val samples: {len(val_loader.dataset)}', 'white'))
		
		# 学習
		print(colored('\nStarting training...', 'cyan', attrs=['bold']))
		
		best_val_mae = float('inf')
		history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
		
		for epoch in range(self.cfg.gru_epochs):
			# Train
			train_metrics = self.train_epoch(train_loader)
			
			# Evaluate
			val_metrics = self.evaluate(val_loader)
			
			# Scheduler
			self.scheduler.step()
			
			# 記録
			history['train_loss'].append(train_metrics['loss'])
			history['train_mae'].append(train_metrics['mae'])
			history['val_loss'].append(val_metrics['loss'])
			history['val_mae'].append(val_metrics['mae'])
			
			# ログ
			if (epoch + 1) % self.cfg.log_interval == 0:
				print(colored(f'Epoch {epoch+1}/{self.cfg.gru_epochs}:', 'yellow'))
				print(f'  Train Loss: {train_metrics["loss"]:.4f}, MAE: {train_metrics["mae"]:.4f}')
				print(f'  Val   Loss: {val_metrics["loss"]:.4f}, MAE: {val_metrics["mae"]:.4f}')
			
			# ベストモデルを保存
			if val_metrics['mae'] < best_val_mae:
				best_val_mae = val_metrics['mae']
				save_path = Path(self.cfg.work_dir) / 'best_gru.pt'
				torch.save({
					'estimator_state_dict': self.estimator.state_dict(),
					'epoch': epoch,
					'val_mae': best_val_mae,
					'cfg': self.cfg,
				}, save_path)
				print(colored(f'  ✓ Saved best model (MAE: {best_val_mae:.4f})', 'green'))
		
		# 最終評価
		print(colored('\n=== Final Evaluation ===', 'cyan', attrs=['bold']))
		val_metrics = self.evaluate(val_loader)
		print(colored(f'Best Val MAE: {best_val_mae:.4f}', 'green'))
		print(colored(f'Final Val MAE: {val_metrics["mae"]:.4f}', 'white'))
		
		# 可視化
		self.plot_results(history, val_metrics)
		
		return history, val_metrics
	
	def plot_results(self, history, val_metrics):
		"""結果の可視化"""
		save_dir = Path(self.cfg.work_dir)
		save_dir.mkdir(parents=True, exist_ok=True)
		
		# 学習曲線
		fig, axes = plt.subplots(1, 2, figsize=(12, 4))
		
		axes[0].plot(history['train_loss'], label='Train', alpha=0.7)
		axes[0].plot(history['val_loss'], label='Val', alpha=0.7)
		axes[0].set_xlabel('Epoch')
		axes[0].set_ylabel('Loss')
		axes[0].set_title('Training Curve (Loss)')
		axes[0].legend()
		axes[0].grid(True, alpha=0.3)
		
		axes[1].plot(history['train_mae'], label='Train', alpha=0.7)
		axes[1].plot(history['val_mae'], label='Val', alpha=0.7)
		axes[1].set_xlabel('Epoch')
		axes[1].set_ylabel('MAE')
		axes[1].set_title('Training Curve (MAE)')
		axes[1].legend()
		axes[1].grid(True, alpha=0.3)
		
		plt.tight_layout()
		plt.savefig(save_dir / 'gru_training_curve.png', dpi=300)
		print(colored(f'✓ Saved: {save_dir / "gru_training_curve.png"}', 'green'))
		
		# 予測 vs 真値
		predictions = val_metrics['predictions']
		targets = val_metrics['targets']
		
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.scatter(targets[:, 0], predictions[:, 0], alpha=0.5, s=10)
		ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 
		        'r--', linewidth=2, label='Perfect prediction')
		ax.set_xlabel('True c_phys')
		ax.set_ylabel('Predicted c_phys')
		ax.set_title(f'GRU Prediction vs Ground Truth (MAE: {val_metrics["mae"]:.4f})')
		ax.legend()
		ax.grid(True, alpha=0.3)
		ax.set_aspect('equal')
		
		plt.tight_layout()
		plt.savefig(save_dir / 'gru_prediction_vs_truth.png', dpi=300)
		print(colored(f'✓ Saved: {save_dir / "gru_prediction_vs_truth.png"}', 'green'))


@hydra.main(config_name='config_gru_offline', config_path='.')
def train(cfg: dict):
	"""
	GRU推定器のオフライン学習。
	
	使用例:
		$ python train_gru_offline.py task=pendulum-swingup-randomized
	"""
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	
	print(colored('='*70, 'yellow', attrs=['bold']))
	print(colored('GRU Offline Training', 'yellow', attrs=['bold']))
	print(colored('='*70, 'yellow', attrs=['bold']))
	print(colored('Task:', 'cyan'), cfg.task)
	print(colored('Work dir:', 'cyan'), cfg.work_dir)
	
	trainer = GRUOfflineTrainer(cfg)
	history, val_metrics = trainer.train()
	
	print(colored('\n✓ Training completed successfully!', 'green', attrs=['bold']))


if __name__ == '__main__':
	train()

