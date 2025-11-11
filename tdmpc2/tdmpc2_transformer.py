"""Transformer-based TD-MPC2 Agent with In-Context Learning

TransformerWorldModelを使用したTD-MPC2エージェント。
履歴から物理法則を推論し、Domain Randomizationされた環境に適応。
"""

import dataclasses
import numpy as np
import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.transformer_world_model import TransformerWorldModel
from common.context_manager import ContextManager
from common.trajectory_buffer import TrajectoryBuffer
from tensordict import TensorDict


class TDMPC2Transformer(torch.nn.Module):
	"""Transformer-based TD-MPC2 Agent
	
	主な変更点:
	- WorldModel → TransformerWorldModel
	- Buffer → TrajectoryBuffer
	- ContextManager追加
	- プランニング時に履歴を活用
	"""
	
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		
		# Transformer World Model
		self.model = TransformerWorldModel(cfg).to(self.device)
		
		# Context Manager（履歴管理）
		self.context_manager = ContextManager(
			context_length=getattr(cfg, 'context_length', 50),
			latent_dim=cfg.latent_dim,
			action_dim=cfg.action_dim,
			device=self.device
		)
		
		# Optimizer
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model.state_emb.parameters()},
			{'params': self.model.action_emb.parameters()},
			{'params': self.model.pos_enc.parameters()},
			{'params': [p for block in self.model.blocks for p in block.parameters()]},
			{'params': self.model.ln_f.parameters()},
			{'params': self.model._dynamics_head.parameters()},
			{'params': self.model._reward_head.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
		], lr=self.cfg.lr, capturable=True)
		
		self.pi_optim = torch.optim.Adam(
			self.model._policy_head.parameters(),
			lr=self.cfg.lr,
			eps=1e-5,
			capturable=True
		)
		
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20)
		
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		print('Context length:', self.context_manager.context_length)
		
		self.register_buffer(
			'_prev_mean',
			torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		)
		
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")
	
	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is None:
			_plan_val = torch.compile(self._plan, mode='reduce-overhead') if self.cfg.compile else self._plan
			setattr(self, "_plan_val", _plan_val)
		return _plan_val
	
	def _get_discount(self, episode_length):
		"""Compute discount factor for a given episode length"""
		return 1 - (1 / episode_length)
	
	def save(self, fp):
		"""Save agent state"""
		from omegaconf import OmegaConf
		
		cfg_obj = self.cfg
		if OmegaConf.is_config(cfg_obj):
			cfg_dict = OmegaConf.to_container(cfg_obj, resolve=True)
		elif dataclasses.is_dataclass(cfg_obj):
			cfg_dict = dataclasses.asdict(cfg_obj)
		elif isinstance(cfg_obj, dict):
			cfg_dict = dict(cfg_obj)
		else:
			cfg_dict = dict(getattr(cfg_obj, "__dict__", {}))
		
		torch.save({
			'model': self.model.state_dict(),
			'optim': self.optim.state_dict(),
			'pi_optim': self.pi_optim.state_dict(),
			'scale': self.scale.state_dict(),
			'cfg': cfg_dict
		}, fp)
		print(f'Saved agent checkpoint to {fp}')
	
	def load(self, fp):
		"""Load agent state"""
		state = torch.load(fp, map_location='cuda:0')
		self.model.load_state_dict(state['model'])
		self.optim.load_state_dict(state['optim'])
		self.pi_optim.load_state_dict(state['pi_optim'])
		self.scale.load_state_dict(state['scale'])
		print(f'Loaded agent checkpoint from {fp}')
	
	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""Select an action using MPPI planning
		
		Args:
			obs: observation
			t0: True if start of episode (reset context)
			eval_mode: True for evaluation (deterministic)
			task: task ID for multi-task
			
		Returns:
			action: selected action
		"""
		# エピソード開始時にコンテキストをリセット
		if t0:
			self.context_manager.reset()
			self.model.reset_cache()
			self._prev_mean.zero_()
		
		# 観測を正規化してエンコード
		obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
		if self.cfg.obs == 'rgb':
			obs = obs / 255.0 - 0.5
		
		z = self.model.encode(obs, task)
		
		# プランニング
		if eval_mode:
			action = self.plan(z, task, eval_mode=True)
		else:
			action = self.plan(z, task, eval_mode=False)
		
		# コンテキストに追加（次のステップで使用）
		action_for_context = action.squeeze(0)
		self.context_manager.append(z.squeeze(0), action_for_context)
		
		# CPUに移動してから返す（環境wrapperがnumpy変換を行う）
		return action_for_context.cpu()
	
	@torch.no_grad()
	def _plan(self, z, task, eval_mode=False):
		"""Trajectory sampling planner with MPPI-style updates."""
		device = self.device
		z = z.to(device)
		horizon = self.cfg.horizon
		num_samples = getattr(self.cfg, 'num_eval_trajs', 1) if eval_mode else self.cfg.num_samples
		num_samples = max(1, num_samples)

		if self.context_manager.is_empty():
			z_base = z.unsqueeze(1).repeat(num_samples, 1, 1)
			a_base = torch.zeros(num_samples, 1, self.cfg.action_dim, device=device)
		else:
			z_ctx, a_ctx = self.context_manager.get_context(batch_size=num_samples)
			z_ctx = z_ctx.to(device)
			a_ctx = a_ctx.to(device)
			z_base = torch.cat([z_ctx, z.unsqueeze(1).repeat(num_samples, 1, 1)], dim=1)
			a_base = torch.cat(
				[a_ctx, torch.zeros(num_samples, 1, self.cfg.action_dim, device=device)],
				dim=1
			)

		max_ctx_tokens = self.context_manager.context_length
		if z_base.shape[1] > max_ctx_tokens:
			z_base = z_base[:, -max_ctx_tokens:]
			a_base = a_base[:, -max_ctx_tokens:]

		mean = torch.zeros(horizon, self.cfg.action_dim, device=device, dtype=torch.float32)
		std = torch.full(
			(horizon, self.cfg.action_dim),
			float(self.cfg.max_std),
			device=device,
			dtype=torch.float32,
		)
		if self.cfg.iterations > 0 and not self.context_manager.is_empty():
			mean[:-1] = self._prev_mean[1:]

		if eval_mode:
			std = std.clamp_(max=self.cfg.min_std)

		if self.cfg.multitask:
			if isinstance(task, torch.Tensor):
				discount_scalar = self.discount[task.long()].mean().item()
			elif task is not None:
				discount_scalar = float(self.discount[int(task)].item())
			else:
				discount_scalar = float(self.discount.mean().item())
		else:
			discount_scalar = float(self.discount)

		for _ in range(self.cfg.iterations):
			noise = torch.randn(horizon, num_samples, self.cfg.action_dim, device=device)
			actions = mean.unsqueeze(1) + std.unsqueeze(1) * noise
			actions = actions.clamp(-1, 1)

			returns = torch.zeros(num_samples, device=device)
			discount_acc = 1.0
			z_hist = z_base.clone()
			a_hist = a_base.clone()

			for t in range(horizon):
				a_hist[:, -1] = actions[t]
				reward_logits = self.model.reward(z_hist, a_hist, task)
				reward = math.two_hot_inv(reward_logits, self.cfg).squeeze(-1)
				returns += discount_acc * reward

				z_next = self.model.next(z_hist, a_hist, task)
				z_hist = torch.cat([z_hist, z_next.unsqueeze(1)], dim=1)
				a_hist = torch.cat(
					[a_hist, torch.zeros(num_samples, 1, self.cfg.action_dim, device=device)],
					dim=1
				)

				if z_hist.shape[1] > max_ctx_tokens:
					z_hist = z_hist[:, -max_ctx_tokens:]
					a_hist = a_hist[:, -max_ctx_tokens:]

				discount_acc *= discount_scalar

			score = torch.softmax(returns / self.cfg.temperature, dim=0)
			score = score.unsqueeze(0).unsqueeze(-1)
			mean = torch.sum(score * actions, dim=1)
			var = torch.sum(score * (actions - mean.unsqueeze(1)) ** 2, dim=1)
			std = torch.clamp(var.sqrt(), self.cfg.min_std, self.cfg.max_std)

			if self.cfg.multitask and task is not None:
				if isinstance(task, torch.Tensor):
					task_idx = int(task.flatten()[0].item())
				else:
					task_idx = int(task)
				mask = self.model._action_masks[task_idx]
				mean = mean * mask
				std = std * mask

		action = mean[0].unsqueeze(0)
		if not eval_mode:
			action = action + std[0].unsqueeze(0) * torch.randn_like(action)
		action = action.clamp(-1, 1)

		momentum = getattr(self.cfg, 'momentum', 0.0)
		action = (1 - momentum) * action + momentum * self._prev_mean[0:1]
		new_prev_mean = torch.cat([action, self._prev_mean[:-1]])
		self._prev_mean.copy_(new_prev_mean)

		return action

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task, obs, action):
		"""Compute TD target
		
		Args:
			next_z: (horizon, batch_size, latent_dim)
			reward: (horizon, batch_size, 1)
			terminated: (horizon, batch_size, 1)
			obs: (seq_len, batch_size, obs_dim) - 履歴用
			action: (horizon, batch_size, action_dim) - アクション履歴
		"""
		# 初期状態をエンコード
		z_0 = self.model.encode(obs[0], task)  # (batch_size, latent_dim)
		
		# next_zの各タイムステップでTD targetを計算
		horizon, batch_size = next_z.shape[0], next_z.shape[1]
		max_ctx_tokens = self.context_manager.context_length
		td_targets = []
		
		for t in range(horizon):
			# TD target計算: r_t + γ * Q(z_{t+1}, π(z_{t+1}))
			# z_{t+1} = next_z[t]での履歴を構築
			# z_history: [z_0, next_z[0], ..., next_z[t]] = [z_0, z_1, ..., z_{t+1}]
			if t == 0:
				z_history = torch.stack([z_0, next_z[0]], dim=0)  # (2, batch, latent_dim)
			else:
				z_history = torch.stack([z_0] + [next_z[i] for i in range(t+1)], dim=0)
			z_history = z_history.permute(1, 0, 2)  # (batch, t+2, latent_dim)
			
			# Policy prior sampling with action context
			a_history = action[:t+1].permute(1, 0, 2)
			padding = torch.zeros(batch_size, 1, self.cfg.action_dim, device=action.device, dtype=action.dtype)
			a_for_pi = torch.cat([a_history, padding], dim=1)
			if z_history.shape[1] > max_ctx_tokens:
				z_history_pi = z_history[:, -max_ctx_tokens:]
				a_for_pi = a_for_pi[:, -max_ctx_tokens:]
			else:
				z_history_pi = z_history
			pi_action, _ = self.model.pi(z_history_pi, task, a_for_pi)
			a_for_q = torch.cat([a_history, pi_action.unsqueeze(1)], dim=1)
			if z_history.shape[1] > max_ctx_tokens:
				z_history_q = z_history[:, -max_ctx_tokens:]
				a_for_q = a_for_q[:, -max_ctx_tokens:]
			else:
				z_history_q = z_history
			Q_value = self.model.Q(z_history_q, a_for_q, task, return_type='min', target=True)
			
			discount = self.discount
			if self.cfg.multitask and isinstance(task, torch.Tensor):
				discount = self.discount[task]
			
			# TD target
			td_target = reward[t] + discount * (1-terminated[t]) * Q_value
			td_targets.append(td_target)
		
		return torch.stack(td_targets, dim=0)
	
	def _update(self, obs, action, reward, terminated, task=None):
		"""Update world model and policy
		
		Args:
			obs: (seq_len, batch_size, obs_dim)
			action: (seq_len-1, batch_size, action_dim)
			reward: (seq_len-1, batch_size, 1)
			terminated: (seq_len-1, batch_size, 1)
			task: task ID
		"""
		# TD targetsを計算
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task, obs, action)
		
		# 訓練モード
		self.model.train()
		
		# Latent rollout with context
		seq_len = obs.shape[0]
		horizon = seq_len - 1
		batch_size = obs.shape[1]
		
		# リストで保存してからstackする（inplace操作を避ける）
		zs_list = []
		z = self.model.encode(obs[0], task)
		zs_list.append(z)
		
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			# 履歴を使った予測
			z_history = torch.stack(zs_list, dim=0).permute(1, 0, 2)  # (batch, t+1, latent_dim)
			a_history = action[:t+1].permute(1, 0, 2) if t > 0 else _action.unsqueeze(1)
			
			z = self.model.next(z_history, a_history, task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs_list.append(z)
		
		# stackしてテンソルに変換
		zs = torch.stack(zs_list, dim=0)
		
		# Predictions
		_zs = zs[:-1]
		
		# 各タイムステップでQ, rewardを予測（履歴を含めて）
		qs_list = []
		reward_preds_list = []
		
		for t in range(horizon):
			# 時刻tでの予測: z_historyは[z_0, ..., z_t]、a_historyは[a_0, ..., a_t]
			# z_tとa_tのペアでreward/Qを予測する
			z_history = zs[:t+1].permute(1, 0, 2)  # (batch, t+1, latent_dim)
			a_history = action[:t+1].permute(1, 0, 2)  # (batch, t+1, action_dim)
			
			qs = self.model.Q(z_history, a_history, task, return_type='all')
			reward_pred = self.model.reward(z_history, a_history, task)
			
			qs_list.append(qs)
			reward_preds_list.append(reward_pred)
		
		# Lossの計算
		reward_loss, value_loss = 0, 0
		for t, (rew_pred, rew, td_target, qs) in enumerate(
			zip(reward_preds_list, reward.unbind(0), td_targets.unbind(0), qs_list)
		):
			reward_loss = reward_loss + math.soft_ce(rew_pred, rew, self.cfg).mean() * self.cfg.rho**t
			for q in qs.unbind(0):
				value_loss = value_loss + math.soft_ce(q, td_target, self.cfg).mean() * self.cfg.rho**t
		
		consistency_loss = consistency_loss / horizon
		reward_loss = reward_loss / horizon
		value_loss = value_loss / (horizon * self.cfg.num_q)
		
		# Termination loss
		if self.cfg.episodic:
			termination_preds = []
			for t in range(1, seq_len):
				z_t = zs[t]
				term_pred = self.model.termination(z_t, task, unnormalized=True)
				termination_preds.append(term_pred)
			termination_pred = torch.stack(termination_preds, dim=0)  # (horizon, batch, 1)
			termination_loss = F.binary_cross_entropy_with_logits(
				termination_pred.squeeze(-1),
				terminated.squeeze(-1)
			)
		else:
			termination_loss = 0.
		
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)
		
		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(
			self.model.parameters(),
			self.cfg.grad_clip_norm, error_if_nonfinite=False
		)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)
		self.model.eval()
		
		# Policy update
		self.model.train()
		pi_loss = self.update_pi(zs[:-1].detach(), action.detach(), task)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.eval()
		
		# Soft update target Q-networks
		self.model.soft_update_target_Q()
		
		return TensorDict({
			'consistency_loss': consistency_loss,
			'reward_loss': reward_loss,
			'value_loss': value_loss,
			'termination_loss': termination_loss if self.cfg.episodic else 0.,
			'pi_loss': pi_loss,
			'total_loss': total_loss,
			'grad_norm': grad_norm,
		})
	
	def update_pi(self, zs, actions, task):
		"""Update policy prior using transformer context.

		Args:
			zs: (horizon+1, batch_size, latent_dim)
			actions: (horizon, batch_size, action_dim)
		"""
		horizon = actions.shape[0]
		pi_loss = 0
		max_ctx_tokens = self.context_manager.context_length
		for t in range(horizon):
			z_history = zs[:t+1].permute(1, 0, 2)
			batch_size = z_history.shape[0]
			if t > 0:
				a_history = actions[:t].permute(1, 0, 2)
			else:
				a_history = actions.new_zeros((batch_size, 0, self.cfg.action_dim))
			padding = torch.zeros(batch_size, 1, self.cfg.action_dim, device=actions.device, dtype=actions.dtype)
			a_for_pi = torch.cat([a_history, padding], dim=1)
			if z_history.shape[1] > max_ctx_tokens:
				z_history_pi = z_history[:, -max_ctx_tokens:]
				a_for_pi = a_for_pi[:, -max_ctx_tokens:]
			else:
				z_history_pi = z_history
			pi_action, info = self.model.pi(z_history_pi, task, a_for_pi)
			a_for_q = torch.cat([a_history, pi_action.unsqueeze(1)], dim=1)
			if z_history.shape[1] > max_ctx_tokens:
				z_history_q = z_history[:, -max_ctx_tokens:]
				a_for_q = a_for_q[:, -max_ctx_tokens:]
			else:
				z_history_q = z_history
			Q = self.model.Q(z_history_q, a_for_q, task, return_type='min', detach=True)
			Q = self.scale(Q, update=(t == 0))
			pi_loss = pi_loss + (-Q.mean() - self.cfg.entropy_coef * info['entropy'].mean()) * self.cfg.rho**t
		return pi_loss / horizon
	
	def update(self, buffer):
		"""Sample a batch and update the agent"""
		obs, action, reward, terminated, task = buffer.sample()
		return self._update(obs, action, reward, terminated, task)
