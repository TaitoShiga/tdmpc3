"""
Oracle用WorldModel

真の物理パラメータ c_phys を条件入力として受け取るWorldModel。

全てのネットワーク（dynamics, reward, Q, pi, termination）に
物理パラメータを注入することで、完璧な物理情報を持つ場合の
理論的上限（Model O）を検証する。
"""
from copy import deepcopy

import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class OracleWorldModel(nn.Module):
	"""
	物理パラメータ条件付きWorldModel。
	
	標準のWorldModelとの違い:
	- 全ネットワークの入力に c_phys_dim を追加
	- c_phys_emb() メソッドで物理パラメータを埋め込み
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
		
		# Encoder（物理パラメータの影響は受けない）
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
		# Paramsの作成
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)
		
		# Modulesの作成
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)
		
		# Paramsの割り当て
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params
	
	def __repr__(self):
		repr = 'Oracle World Model (with c_phys conditioning)\n'
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
		"""
		trainメソッドをオーバーライドして、target Q-networksをeval modeに保つ。
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self
	
	def soft_update_target_Q(self):
		"""
		Polyak averagingでtarget Q-networksをソフトアップデート。
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)
	
	def task_emb(self, x, task):
		"""
		Multi-task実験用の連続タスク埋め込み。
		タスクID `task` のタスク埋め込みを取得し、入力 `x` と連結。
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)
	
	def c_phys_emb(self, x, c_phys):
		"""
		物理パラメータ埋め込み。
		
		c_phys を入力 x と連結。
		
		Args:
			x: 入力テンソル (batch_size, dim) or (horizon, batch_size, dim)
			c_phys: 物理パラメータ (batch_size, c_phys_dim)
		
		Returns:
			torch.cat([x, c_phys], dim=-1)
		"""
		if c_phys is None:
			# c_physがNoneの場合、ゼロで埋める（下位互換性）
			if x.ndim == 3:
				c_phys = torch.zeros(x.shape[1], self.c_phys_dim, device=x.device, dtype=x.dtype)
			else:
				c_phys = torch.zeros(x.shape[0], self.c_phys_dim, device=x.device, dtype=x.dtype)
		
		# c_physの次元を調整
		if x.ndim == 3 and c_phys.ndim == 2:
			# x: (horizon, batch, dim), c_phys: (batch, c_phys_dim)
			# -> c_phys: (horizon, batch, c_phys_dim)
			c_phys = c_phys.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif x.ndim == 2 and c_phys.ndim == 2:
			# x: (batch, dim), c_phys: (batch, c_phys_dim)
			# そのまま連結可能
			pass
		elif c_phys.shape[0] == 1 and x.shape[0] > 1:
			# c_phys: (1, c_phys_dim) を batch サイズに拡張
			if x.ndim == 3:
				c_phys = c_phys.unsqueeze(0).repeat(x.shape[0], x.shape[1], 1)
			else:
				c_phys = c_phys.repeat(x.shape[0], 1)
		
		return torch.cat([x, c_phys], dim=-1)
	
	def encode(self, obs, task):
		"""
		観測を潜在表現にエンコード。
		
		Note: Encoderは物理パラメータの影響を受けない
		      （観測は既に物理法則の影響を受けた結果なため）
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)
	
	def next(self, z, a, task, c_phys):
		"""
		現在の潜在状態とアクションから次の潜在状態を予測。
		
		Args:
			z: 現在の潜在状態 (batch, latent_dim)
			a: アクション (batch, action_dim)
			task: タスクID
			c_phys: 物理パラメータ (batch, c_phys_dim)
		
		Returns:
			z': 次の潜在状態 (batch, latent_dim)
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		z = self.c_phys_emb(z, c_phys)
		return self._dynamics(z)
	
	def reward(self, z, a, task, c_phys):
		"""
		即時報酬を予測。
		
		Args:
			z: 潜在状態 (batch, latent_dim)
			a: アクション (batch, action_dim)
			task: タスクID
			c_phys: 物理パラメータ (batch, c_phys_dim)
		
		Returns:
			r: 報酬のTwo-hot表現 (batch, num_bins)
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		z = self.c_phys_emb(z, c_phys)
		return self._reward(z)
	
	def termination(self, z, task, c_phys, unnormalized=False):
		"""
		終了信号を予測。
		
		Args:
			z: 潜在状態 (batch, latent_dim)
			task: タスクID
			c_phys: 物理パラメータ (batch, c_phys_dim)
			unnormalized: Trueの場合、sigmoidを適用しない
		
		Returns:
			p(terminated): 終了確率 (batch, 1)
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = self.c_phys_emb(z, c_phys)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))
	
	def pi(self, z, task, c_phys):
		"""
		ポリシー事前分布からアクションをサンプル。
		
		Args:
			z: 潜在状態 (batch, latent_dim)
			task: タスクID
			c_phys: 物理パラメータ (batch, c_phys_dim)
		
		Returns:
			action: サンプルされたアクション (batch, action_dim)
			info: 追加情報（mean, log_std, entropy等）
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = self.c_phys_emb(z, c_phys)
		
		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)
		
		if self.cfg.multitask:  # 未使用のアクション次元をマスク
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else:  # マスクなし
			action_dims = None
		
		log_prob = math.gaussian_logprob(eps, log_std)
		
		# アクション次元でログ確率をスケール
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size
		
		# Reparameterization trick
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
		"""
		状態-アクション価値を予測。
		
		Args:
			z: 潜在状態 (batch, latent_dim)
			a: アクション (batch, action_dim)
			task: タスクID
			c_phys: 物理パラメータ (batch, c_phys_dim)
			return_type: 'min', 'avg', 'all' のいずれか
			target: target Q-networkを使用するか
			detach: detach Q-networkを使用するか
		
		Returns:
			Q値
		"""
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

