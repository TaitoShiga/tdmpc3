"""
Model C用WorldModel

GRU推定器 + 物理パラメータ条件付きWorldModelを統合。

【重要】勾配分離の実装:
- GRU推定器: L_aux（推定損失）のみで更新
- プランナー: L_TD-MPC2（制御損失）のみで更新
- プランナーに渡すc_physは.detach()して勾配を切る
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import layers, math, init
from common.physics_estimator import create_physics_estimator
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModelC(nn.Module):
	"""
	Model C用のWorldModel。
	
	アーキテクチャ:
	  1. GRU推定器: 履歴 -> c_phys
	  2. Oracle WorldModel: (z, a, task, c_phys) -> 予測
	
	学習:
	  - GRU: L_aux = MSE(c_phys_pred, c_phys_true)
	  - WorldModel: L_TD-MPC2 (consistency, reward, value, ...)
	  - 勾配分離: プランナーに渡すc_physはdetach()
	"""
	
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		
		# 物理パラメータの次元
		self.c_phys_dim = getattr(cfg, 'c_phys_dim', 1)
		
		# Multi-task設定
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		
		# 【フェーズ1】GRU推定器
		self._physics_estimator = create_physics_estimator(
			estimator_type=getattr(cfg, 'estimator_type', 'gru'),
			obs_dim=cfg.obs_shape[cfg.obs][0] if isinstance(cfg.obs_shape, dict) else cfg.obs_shape[0],
			action_dim=cfg.action_dim,
			c_phys_dim=self.c_phys_dim,
			hidden_dim=getattr(cfg, 'gru_hidden_dim', 256),
			num_layers=getattr(cfg, 'gru_num_layers', 2),
			dropout=getattr(cfg, 'gru_dropout', 0.1),
			context_length=getattr(cfg, 'context_length', 50),
		)
		
		# 【フェーズ2】Oracle WorldModel（物理パラメータ条件付き）
		# Encoder
		self._encoder = layers.enc(cfg)
		
		# Dynamics: (z, a, task, c_phys) -> z'
		self._dynamics = layers.mlp(
			cfg.latent_dim + cfg.action_dim + cfg.task_dim + self.c_phys_dim,
			2*[cfg.mlp_dim],
			cfg.latent_dim,
			act=layers.SimNorm(cfg)
		)
		
		# Reward: (z, a, task, c_phys) -> r
		self._reward = layers.mlp(
			cfg.latent_dim + cfg.action_dim + cfg.task_dim + self.c_phys_dim,
			2*[cfg.mlp_dim],
			max(cfg.num_bins, 1)
		)
		
		# Termination: (z, task, c_phys) -> p(terminated)
		if cfg.episodic:
			self._termination = layers.mlp(
				cfg.latent_dim + cfg.task_dim + self.c_phys_dim,
				2*[cfg.mlp_dim],
				1
			)
		else:
			self._termination = None
		
		# Policy prior: (z, task, c_phys) -> (mean, log_std)
		self._pi = layers.mlp(
			cfg.latent_dim + cfg.task_dim + self.c_phys_dim,
			2*[cfg.mlp_dim],
			2*cfg.action_dim
		)
		
		# Q-ensemble: (z, a, task, c_phys) -> Q
		self._Qs = layers.Ensemble([
			layers.mlp(
				cfg.latent_dim + cfg.action_dim + cfg.task_dim + self.c_phys_dim,
				2*[cfg.mlp_dim],
				max(cfg.num_bins, 1),
				dropout=cfg.dropout
			) for _ in range(cfg.num_q)
		])
		
		# 重みの初期化
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])
		
		# ログ標準偏差の範囲
		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		
		self.init()
	
	def init(self):
		"""Q-networkのtarget/detach版を作成"""
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)
		
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)
		
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params
	
	def __repr__(self):
		repr = 'Model C World Model (GRU + Physics-Conditioned)\n'
		repr += f"Physics Estimator: {self._physics_estimator}\n"
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += f"c_phys_dim: {self.c_phys_dim}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr
	
	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
	
	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self
	
	def train(self, mode=True):
		"""trainメソッドをオーバーライド"""
		super().train(mode)
		self._target_Qs.train(False)
		return self
	
	def soft_update_target_Q(self):
		"""Polyak averagingでtarget Q-networksをソフトアップデート"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)
	
	def task_emb(self, x, task):
		"""Multi-task実験用の連続タスク埋め込み"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)
	
	def c_phys_emb(self, x, c_phys):
		"""物理パラメータ埋め込み"""
		if c_phys is None:
			if x.ndim == 3:
				c_phys = torch.zeros(x.shape[1], self.c_phys_dim, device=x.device, dtype=x.dtype)
			else:
				c_phys = torch.zeros(x.shape[0], self.c_phys_dim, device=x.device, dtype=x.dtype)
		
		if x.ndim == 3 and c_phys.ndim == 2:
			c_phys = c_phys.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif x.ndim == 2 and c_phys.ndim == 2:
			pass
		elif c_phys.shape[0] == 1 and x.shape[0] > 1:
			if x.ndim == 3:
				c_phys = c_phys.unsqueeze(0).repeat(x.shape[0], x.shape[1], 1)
			else:
				c_phys = c_phys.repeat(x.shape[0], 1)
		
		return torch.cat([x, c_phys], dim=-1)
	
	def estimate_physics(self, obs_seq, action_seq):
		"""
		【フェーズ1】GRU推定器で物理パラメータを推定。
		
		Args:
			obs_seq: 観測系列 (batch, seq_len, obs_dim)
			action_seq: 行動系列 (batch, seq_len, action_dim)
		
		Returns:
			c_phys_pred: 推定された物理パラメータ (batch, c_phys_dim)
		"""
		c_phys_pred, _ = self._physics_estimator(obs_seq, action_seq)
		return c_phys_pred
	
	def compute_physics_estimation_loss(self, obs_seq, action_seq, c_phys_true):
		"""
		GRU推定器の損失 L_aux を計算。
		
		この損失はGRU推定器のみを更新する。
		"""
		loss, info = self._physics_estimator.compute_loss(obs_seq, action_seq, c_phys_true)
		return loss, info
	
	def encode(self, obs, task):
		"""観測を潜在表現にエンコード"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)
	
	def next(self, z, a, task, c_phys):
		"""
		【フェーズ2】次の潜在状態を予測（c_phys条件付き）。
		
		🔑 重要: c_physはdetach()されている必要がある（呼び出し側で実施）
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		z = self.c_phys_emb(z, c_phys)
		return self._dynamics(z)
	
	def reward(self, z, a, task, c_phys):
		"""報酬を予測（c_phys条件付き）"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		z = self.c_phys_emb(z, c_phys)
		return self._reward(z)
	
	def termination(self, z, task, c_phys, unnormalized=False):
		"""終了信号を予測（c_phys条件付き）"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = self.c_phys_emb(z, c_phys)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))
	
	def pi(self, z, task, c_phys):
		"""ポリシー事前分布からアクションをサンプル（c_phys条件付き）"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = self.c_phys_emb(z, c_phys)
		
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)
		
		if self.cfg.multitask:
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else:
			action_dims = None
		
		log_prob = math.gaussian_logprob(eps, log_std)
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size
		
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)
		
		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info
	
	def Q(self, z, a, task, c_phys, return_type='min', target=False, detach=False):
		"""Q値を予測（c_phys条件付き）"""
		assert return_type in {'min', 'avg', 'all'}
		
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		
		z = torch.cat([z, a], dim=-1)
		z = self.c_phys_emb(z, c_phys)
		
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)
		
		if return_type == 'all':
			return out
		
		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2

