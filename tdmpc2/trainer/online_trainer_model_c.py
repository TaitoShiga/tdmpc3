"""
Model C用オンラインTrainer

履歴管理 + 物理パラメータ収集を行うTrainer。

特徴:
- エピソードごとに履歴（obs, action）を収集
- GRU推定用のウィンドウを切り出してバッファに保存
- エージェントは自動的に履歴から物理パラメータを推定
"""
from time import time
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainerModelC(Trainer):
	"""
	Model C版のシングルタスクオンライン学習Trainer。
	
	履歴管理と物理パラメータ収集を行う。
	"""
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		
		# 履歴バッファ
		self.context_length = getattr(self.cfg, 'context_length', 50)
		self._reset_episode_data()
	
	def _reset_episode_data(self):
		"""エピソードデータをリセット"""
		self._obs_history = []
		self._action_history = []
	
	def _get_history_window(self, t):
		"""
		現在のタイムステップtにおける履歴ウィンドウを取得。
		
		Args:
			t: 現在のタイムステップ
		
		Returns:
			obs_window: (context_length, obs_dim)
			action_window: (context_length, action_dim)
		"""
		# 履歴の長さ
		hist_len = len(self._obs_history)
		
		if hist_len >= self.context_length:
			# 十分な履歴がある場合
			start_idx = hist_len - self.context_length
			obs_window = self._obs_history[start_idx:hist_len]
			action_window = self._action_history[start_idx:hist_len]
		else:
			# 履歴が不十分な場合はゼロパディング
			pad_len = self.context_length - hist_len
			obs_dim = self._obs_history[0].shape[0] if hist_len > 0 else self.env.observation_space.shape[0]
			action_dim = self._action_history[0].shape[0] if hist_len > 0 else self.env.action_space.shape[0]
			
			obs_pad = [torch.zeros(obs_dim) for _ in range(pad_len)]
			action_pad = [torch.zeros(action_dim) for _ in range(pad_len)]
			
			obs_window = obs_pad + self._obs_history
			action_window = action_pad + self._action_history
		
		# Tensorに変換
		obs_window = torch.stack(obs_window)
		action_window = torch.stack(action_window)
		
		return obs_window, action_window
	
	def common_metrics(self):
		"""現在のメトリクスを返す"""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)
	
	def eval(self):
		"""Model Cエージェントを評価"""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			
			# エージェントの履歴をリセット
			self.agent.reset_history()
			
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				
				# エージェントの履歴を更新
				self.agent.update_history(obs, action)
				
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length=np.nanmean(ep_lengths),
		)
	
	def to_td(self, obs, action=None, reward=None, terminated=None, c_phys=None, t=None):
		"""
		新しいステップ用のTensorDictを作成。
		
		Args:
			obs: 観測
			action: アクション
			reward: 報酬
			terminated: 終了フラグ
			c_phys: 物理パラメータ
			t: タイムステップ（履歴ウィンドウの取得用）
		"""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		if c_phys is None:
			c_phys = self.env.current_c_phys
		
		# 履歴ウィンドウを取得
		if t is not None:
			obs_window, action_window = self._get_history_window(t)
		else:
			# エピソード開始時（履歴なし）
			obs_dim = obs.shape[1] if obs.ndim > 1 else obs.shape[0]
			action_dim = action.shape[0]
			obs_window = torch.zeros(self.context_length, obs_dim)
			action_window = torch.zeros(self.context_length, action_dim)
		
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
			c_phys=c_phys.unsqueeze(0).cpu(),
			obs_history=obs_window.unsqueeze(0).cpu(),
			action_history=action_window.unsqueeze(0).cpu(),
			batch_size=(1,)
		)
		return td
	
	def train(self):
		"""Model Cエージェントを学習"""
		train_metrics, done, eval_next = {}, True, False
		last_step_metrics = None
		log_interval = getattr(self.cfg, 'log_interval', 1000)
		
		while self._step <= self.cfg.steps:
			# 定期的にエージェントを評価
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True
			
			# 環境のリセット
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False
				
				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError(
							'Termination detected but you are not in episodic mode. '
							'Set `episodic=true` to enable support for terminations.'
						)
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum().item(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated']
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					train_metrics = {}
					last_step_metrics = None
					self._ep_idx = self.buffer.add(torch.cat(self._tds))
				
				obs = self.env.reset()
				c_phys = self.env.current_c_phys
				
				# エピソードデータをリセット
				self._reset_episode_data()
				self._tds = [self.to_td(obs, c_phys=c_phys, t=0)]
				
				# エージェントの履歴をリセット
				self.agent.reset_history()
			
			# 経験の収集
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			
			# 履歴を更新
			self._obs_history.append(obs.clone())
			self._action_history.append(action.clone())
			
			# エージェントの履歴も更新
			self.agent.update_history(obs, action)
			
			obs, reward, done, info = self.env.step(action)
			
			# TensorDictに追加
			t = len(self._tds)
			self._tds.append(self.to_td(obs, action, reward, info['terminated'], c_phys, t))
			
			# エージェントの更新
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				
				_train_metrics = None
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				
				if _train_metrics is not None:
					step_metrics = {}
					for k, v in _train_metrics.items():
						step_metrics[k] = float(v.item()) if torch.is_tensor(v) else v
					if self.cfg.multitask and 'task' in step_metrics:
						step_metrics.pop('task')
					last_step_metrics = step_metrics
					if self._step % log_interval == 0:
						step_metrics.update(self.common_metrics())
						self.logger.log(step_metrics, 'train')
			
			self._step += 1
		
		if last_step_metrics:
			last_step_metrics.update(self.common_metrics())
			self.logger.log(last_step_metrics, 'train')
		
		self.logger.finish(self.agent)

