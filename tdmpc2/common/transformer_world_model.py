"""Transformer-based World Model for TD-MPC2 with In-Context Learning

Transformerベースのworld modelにより、履歴から現在の物理法則を推論し、
Domain Randomizationされた環境に適応する。
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import layers, math, init
from common.transformer_layers import TransformerBlock, PositionalEncoding
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class TransformerWorldModel(nn.Module):
	"""Transformer-based World Model for In-Context Learning
	
	Architecture:
		Input: (z_{t-L:t}, a_{t-L:t})  - 履歴
		↓
		Token Embedding (state + action)
		↓
		Positional Encoding
		↓
		Transformer Blocks (Causal)
		↓
		Context Vectors c_{t-L:t}
		↓
		Individual Heads: dynamics, reward, Q, policy
		
	Args:
		cfg: Configuration object
			- context_length: 履歴長 (default: 50)
			- transformer_layers: Transformerレイヤー数 (default: 4)
			- transformer_heads: アテンションヘッド数 (default: 8)
			- transformer_dim: Transformer埋め込み次元 (default: 512)
	"""
	
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		
		# Transformerのハイパーパラメータ
		self.context_length = getattr(cfg, 'context_length', 50)
		self.n_layers = getattr(cfg, 'transformer_layers', 4)
		self.n_heads = getattr(cfg, 'transformer_heads', 8)
		self.embed_dim = getattr(cfg, 'transformer_dim', cfg.mlp_dim)
		
		# Multi-task embedding
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		
		# Encoder: obs -> z
		self._encoder = layers.enc(cfg)
		
		# Token embedding: (z, a) -> embedding
		self.state_emb = nn.Linear(cfg.latent_dim, self.embed_dim)
		self.action_emb = nn.Linear(cfg.action_dim, self.embed_dim)
		
		# Positional encoding
		self.pos_enc = PositionalEncoding(self.embed_dim, max_len=self.context_length * 2)
		
		# Transformer blocks
		self.blocks = nn.ModuleList([
			TransformerBlock(
				self.embed_dim,
				self.n_heads,
				mlp_ratio=4.0,
				dropout=cfg.dropout
			)
			for _ in range(self.n_layers)
		])
		
		self.ln_f = nn.LayerNorm(self.embed_dim)
		
		# Individual prediction heads
		self._dynamics_head = nn.Sequential(
			nn.Linear(self.embed_dim + cfg.action_dim, cfg.mlp_dim),
			nn.LayerNorm(cfg.mlp_dim),
			nn.Tanh(),
			nn.Linear(cfg.mlp_dim, cfg.latent_dim),
		)
		
		self._reward_head = nn.Sequential(
			nn.Linear(self.embed_dim + cfg.action_dim, cfg.mlp_dim),
			nn.LayerNorm(cfg.mlp_dim),
			nn.Tanh(),
			nn.Linear(cfg.mlp_dim, max(cfg.num_bins, 1)),
		)
		
		self._policy_head = nn.Sequential(
			nn.Linear(self.embed_dim, cfg.mlp_dim),
			nn.LayerNorm(cfg.mlp_dim),
			nn.Tanh(),
			nn.Linear(cfg.mlp_dim, 2 * cfg.action_dim),
		)
		
		# Q-ensemble (MLPと同じ構造)
		self._Qs = layers.Ensemble([
			layers.mlp(
				self.embed_dim + cfg.action_dim, 
				2*[cfg.mlp_dim], 
				max(cfg.num_bins, 1), 
				dropout=cfg.dropout
			) for _ in range(cfg.num_q)
		])
		
		# Termination head (episodic tasks)
		self._termination = layers.mlp(
			self.embed_dim, 
			2*[cfg.mlp_dim], 
			1
		) if cfg.episodic else None
		
		# 重みの初期化
		self.apply(init.weight_init)
		init.zero_([self._reward_head[-1].weight, self._Qs.params["2", "weight"]])
		
		# K-Vキャッシュ（プランニング高速化用）
		self._kv_cache = None
		
		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()
	
	def init(self):
		"""Q-networksの初期化（TD-MPC2と同じ）"""
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params
	
	def __repr__(self):
		repr_str = 'Transformer-based TD-MPC2 World Model\n'
		repr_str += f"Context Length: {self.context_length}\n"
		repr_str += f"Transformer Layers: {self.n_layers}\n"
		repr_str += f"Attention Heads: {self.n_heads}\n"
		repr_str += f"Embedding Dim: {self.embed_dim}\n"
		modules = ['Encoder', 'Transformer', 'Dynamics Head', 'Reward Head', 
				   'Termination', 'Policy Head', 'Q-functions']
		for i, m in enumerate([self._encoder, self.blocks, self._dynamics_head, 
							   self._reward_head, self._termination, 
							   self._policy_head, self._Qs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr_str += f"{modules[i]}: {m}\n"
		repr_str += "Learnable parameters: {:,}".format(self.total_params)
		return repr_str
	
	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
	
	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self
	
	def train(self, mode=True):
		"""Target Q-networksは常にeval mode"""
		super().train(mode)
		if hasattr(self, '_target_Qs'):
			self._target_Qs.train(False)
		return self
	
	def soft_update_target_Q(self):
		"""Soft-update target Q-networks using Polyak averaging"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)
	
	def task_emb(self, x, task):
		"""Task embedding for multi-task experiments"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)
	
	def encode(self, obs, task):
		"""Encodes an observation into its latent representation"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)
	
	def forward(self, z_history, a_history, task=None, use_cache=False):
		"""Transformer forward pass
		
		Args:
			z_history: (batch, seq_len, latent_dim)
			a_history: (batch, seq_len, action_dim)
			task: task ID (for multi-task)
			use_cache: K-Vキャッシュを使用（プランニング高速化）
			
		Returns:
			context: (batch, seq_len, embed_dim) - 文脈ベクトル
		"""
		B, T, _ = z_history.shape
		
		# Token embedding: state + action
		z_tokens = self.state_emb(z_history)  # (B, T, embed_dim)
		a_tokens = self.action_emb(a_history)  # (B, T, embed_dim)
		tokens = z_tokens + a_tokens
		
		# Positional encoding
		x = self.pos_enc(tokens)
		
		# Transformer blocks with K-V caching
		new_caches = []
		for i, block in enumerate(self.blocks):
			past_kv = self._kv_cache[i] if use_cache and self._kv_cache else None
			x, cache = block(x, use_cache, past_kv)
			if use_cache:
				new_caches.append(cache)
		
		if use_cache:
			self._kv_cache = new_caches
		
		context = self.ln_f(x)
		return context
	
	def reset_cache(self):
		"""K-Vキャッシュをリセット（エピソード開始時）"""
		self._kv_cache = None
	
	def next(self, z, a, task):
		"""Predicts the next latent state
		
		MLPモデルとの互換性のため、シングルステップの予測も可能。
		実際は履歴が必要だが、ここではz, aをシーケンスとして扱う。
		
		Args:
			z: (batch, latent_dim) or (batch, seq_len, latent_dim)
			a: (batch, action_dim) or (batch, seq_len, action_dim)
			task: task ID
			
		Returns:
			z_next: (batch, latent_dim)
		"""
		# シーケンス形式に変換
		if z.ndim == 2:
			z = z.unsqueeze(1)  # (batch, 1, latent_dim)
		if a.ndim == 2:
			a = a.unsqueeze(1)  # (batch, 1, action_dim)
		
		# Transformer forward
		context = self.forward(z, a, task, use_cache=False)
		c_t = context[:, -1]  # 最後のタイムステップ
		
		# Dynamics prediction
		# アクションは現在のものを使う（シーケンスの最後）
		a_current = a[:, -1] if a.size(1) > 1 else a.squeeze(1)
		z_next = self._dynamics_head(torch.cat([c_t, a_current], dim=-1))
		
		return z_next
	
	def reward(self, z, a, task):
		"""Predicts instantaneous reward"""
		if z.ndim == 2:
			z = z.unsqueeze(1)
		if a.ndim == 2:
			a = a.unsqueeze(1)
		
		context = self.forward(z, a, task, use_cache=False)
		c_t = context[:, -1]
		a_current = a[:, -1] if a.size(1) > 1 else a.squeeze(1)
		
		return self._reward_head(torch.cat([c_t, a_current], dim=-1))
	
	def termination(self, z, task, unnormalized=False):
		"""Predicts termination signal"""
		if z.ndim == 2:
			z = z.unsqueeze(1)
		
		# Actionは不要なので、ゼロパディング
		a_dummy = torch.zeros(z.size(0), z.size(1), self.cfg.action_dim, device=z.device)
		
		context = self.forward(z, a_dummy, task, use_cache=False)
		c_t = context[:, -1]
		
		if unnormalized:
			return self._termination(c_t)
		return torch.sigmoid(self._termination(c_t))
	
	def pi(self, z, task, a=None):
		"""Samples an action from the policy prior"""
		if z.ndim == 2:
			z = z.unsqueeze(1)
		if a is None:
			a = torch.zeros(z.size(0), z.size(1), self.cfg.action_dim, device=z.device)
		elif a.ndim == 2:
			a = a.unsqueeze(1)

		context = self.forward(z, a, task, use_cache=False)
		c_t = context[:, -1]
		
		# Gaussian policy prior
		mean, log_std = self._policy_head(c_t).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask:  # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else:
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)
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
	
	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""Predict state-action value"""
		assert return_type in {'min', 'avg', 'all'}
		
		if z.ndim == 2:
			z = z.unsqueeze(1)
		if a.ndim == 2:
			a = a.unsqueeze(1)
		
		context = self.forward(z, a, task, use_cache=False)
		c_t = context[:, -1]
		a_current = a[:, -1] if a.size(1) > 1 else a.squeeze(1)
		
		inp = torch.cat([c_t, a_current], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(inp)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2

