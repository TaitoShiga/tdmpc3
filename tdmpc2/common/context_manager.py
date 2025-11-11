"""Context Manager for Transformer-based TD-MPC2

履歴（コンテキスト）を管理し、プランニング時に過去Lステップの(z, a)を保持・更新する。
In-Context Learningに必要な軌跡情報を提供。
"""

import torch
import torch.nn as nn


class ContextManager:
	"""履歴（コンテキスト）を管理するクラス
	
	プランニング時に過去Lステップの(z, a)を保持・更新。
	エピソード開始時にリセット、ステップごとに更新。
	
	Args:
		context_length: 保持する履歴の長さ (L)
		latent_dim: 潜在状態の次元
		action_dim: アクションの次元
		device: torch device
	"""
	def __init__(self, context_length, latent_dim, action_dim, device):
		self.context_length = context_length
		self.latent_dim = latent_dim
		self.action_dim = action_dim
		self.device = device
		
		# 履歴バッファ (FIFO queue)
		self.z_history = torch.zeros(context_length, latent_dim, device=device)
		self.a_history = torch.zeros(context_length, action_dim, device=device)
		self.current_len = 0
		
	def reset(self):
		"""エピソード開始時にリセット"""
		self.z_history.zero_()
		self.a_history.zero_()
		self.current_len = 0
		
	def append(self, z, a):
		"""新しいステップを追加
		
		Args:
			z: (latent_dim,) or (batch, latent_dim)
			a: (action_dim,) or (batch, action_dim)
		"""
		# バッチ次元を削除（必要なら）
		if z.ndim == 2:
			z = z.squeeze(0)
		if a.ndim == 2:
			a = a.squeeze(0)
		
		if self.current_len < self.context_length:
			# バッファがまだ満杯でない
			self.z_history[self.current_len] = z
			self.a_history[self.current_len] = a
			self.current_len += 1
		else:
			# FIFO: 古いデータを削除して新しいデータを追加
			self.z_history = torch.cat([self.z_history[1:], z.unsqueeze(0)])
			self.a_history = torch.cat([self.a_history[1:], a.unsqueeze(0)])
	
	def get_context(self, batch_size=1):
		"""現在のコンテキストを取得
		
		Args:
			batch_size: バッチサイズ（プランニング時）
			
		Returns:
			z_history: (batch_size, seq_len, latent_dim)
			a_history: (batch_size, seq_len, action_dim)
		"""
		seq_len = min(self.current_len, self.context_length)
		
		z_ctx = self.z_history[:seq_len].unsqueeze(0)  # (1, seq_len, latent_dim)
		a_ctx = self.a_history[:seq_len].unsqueeze(0)  # (1, seq_len, action_dim)
		
		# バッチサイズに合わせて複製
		if batch_size > 1:
			z_ctx = z_ctx.expand(batch_size, -1, -1)
			a_ctx = a_ctx.expand(batch_size, -1, -1)
		
		return z_ctx, a_ctx
	
	def get_length(self):
		"""現在の履歴長を取得"""
		return self.current_len
	
	def is_empty(self):
		"""履歴が空かどうか"""
		return self.current_len == 0
	
	def __repr__(self):
		return (
			f"ContextManager(context_length={self.context_length}, "
			f"current_len={self.current_len}, "
			f"latent_dim={self.latent_dim}, "
			f"action_dim={self.action_dim})"
		)


def test_context_manager():
	"""ContextManagerのユニットテスト"""
	print("Testing ContextManager...")
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	context_length = 10
	latent_dim = 64
	action_dim = 1
	
	cm = ContextManager(context_length, latent_dim, action_dim, device)
	
	# 初期状態
	assert cm.is_empty()
	assert cm.get_length() == 0
	
	# データ追加（バッファが満杯になるまで）
	for i in range(15):
		z = torch.randn(latent_dim, device=device) * i
		a = torch.randn(action_dim, device=device) * i
		cm.append(z, a)
	
	# バッファは最大context_lengthまで
	assert cm.get_length() == context_length
	
	# コンテキスト取得
	z_ctx, a_ctx = cm.get_context(batch_size=1)
	assert z_ctx.shape == (1, context_length, latent_dim)
	assert a_ctx.shape == (1, context_length, action_dim)
	
	# バッチサイズ指定
	z_ctx_batch, a_ctx_batch = cm.get_context(batch_size=4)
	assert z_ctx_batch.shape == (4, context_length, latent_dim)
	assert a_ctx_batch.shape == (4, context_length, action_dim)
	
	# リセット
	cm.reset()
	assert cm.is_empty()
	assert cm.get_length() == 0
	
	# 少量のデータでコンテキスト取得
	for i in range(3):
		z = torch.randn(latent_dim, device=device)
		a = torch.randn(action_dim, device=device)
		cm.append(z, a)
	
	z_ctx_short, a_ctx_short = cm.get_context()
	assert z_ctx_short.shape == (1, 3, latent_dim)
	assert a_ctx_short.shape == (1, 3, action_dim)
	
	print("✓ ContextManager tests passed")


if __name__ == "__main__":
	test_context_manager()
	print("\n✅ All tests passed!")

