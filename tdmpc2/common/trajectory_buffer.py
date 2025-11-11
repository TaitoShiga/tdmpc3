"""Trajectory Buffer for Transformer-based TD-MPC2

軌道チャンク（連続Lステップ）をサンプリングするリプレイバッファ。
Transformerが履歴から物理法則を推論するために必要。
"""

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

from common.buffer import Buffer


class TrajectoryBuffer(Buffer):
	"""Transformer用のTrajectory Buffer
	
	通常のBufferと異なり、より長いコンテキスト（context_length）を
	サンプリングする。エピソード内の連続したステップを取得。
	
	Args:
		cfg: Configuration
			- context_length: サンプリングする履歴の長さ (default: 50)
			- horizon: プランニングホライズン
			- batch_size: バッチサイズ
	"""
	
	def __init__(self, cfg):
		# context_lengthを設定（デフォルト50）
		self.context_length = getattr(cfg, 'context_length', 50)
		
		# 親クラスの初期化
		# ただし、samplerは自分で作り直す
		self.cfg = cfg
		self._device = torch.device('cuda:0')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		
		# Transformer用に長いスライスをサンプリング
		# horizon + context_length のチャンクを取得
		self._slice_length = cfg.horizon + self.context_length
		
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		
		# バッチサイズの調整
		self._batch_size = cfg.batch_size * self._slice_length
		self._num_eps = 0
	
	def _prepare_batch(self, td):
		"""Prepare a sampled batch for Transformer training
		
		通常のBufferと異なり、履歴情報も含める。
		
		Args:
			td: TensorDict with shape (slice_length, batch_size)
			
		Returns:
			obs: (slice_length, batch_size, obs_dim)
			action: (slice_length-1, batch_size, action_dim)
			reward: (slice_length-1, batch_size, 1)
			terminated: (slice_length-1, batch_size, 1)
			task: (batch_size,) or None
		"""
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(
			self._device, non_blocking=True
		)
		
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		
		return obs, action, reward, terminated, task
	
	def sample(self):
		"""Sample a batch of trajectory chunks from the buffer
		
		Returns:
			obs: (slice_length, batch_size, obs_dim)
			action: (slice_length-1, batch_size, action_dim)
			reward: (slice_length-1, batch_size, 1)
			terminated: (slice_length-1, batch_size, 1)
			task: (batch_size,) or None
		"""
		# SliceSamplerを使って連続したチャンクをサンプリング
		td = self._buffer.sample().view(-1, self._slice_length).permute(1, 0)
		return self._prepare_batch(td)


def test_trajectory_buffer():
	"""TrajectoryBufferのユニットテスト"""
	print("Testing TrajectoryBuffer...")
	
	# 簡易的なconfigを作成
	from argparse import Namespace
	cfg = Namespace(
		buffer_size=10000,
		steps=10000,
		batch_size=4,
		horizon=10,
		context_length=20,
		multitask=False,
	)
	
	buffer = TrajectoryBuffer(cfg)
	
	# ダミーエピソードを追加
	episode_length = 100
	obs_dim = 5
	action_dim = 1
	
	for ep_idx in range(3):
		td = TensorDict({
			'obs': torch.randn(episode_length, obs_dim),
			'action': torch.randn(episode_length, action_dim),
			'reward': torch.randn(episode_length),
			'terminated': torch.zeros(episode_length),
		}, batch_size=[episode_length])
		
		buffer.add(td)
	
	print(f"Added {buffer.num_eps} episodes")
	
	# サンプリングテスト
	obs, action, reward, terminated, task = buffer.sample()
	
	expected_slice_length = cfg.horizon + cfg.context_length
	print(f"Expected slice length: {expected_slice_length}")
	print(f"Sampled obs shape: {obs.shape}")
	print(f"Sampled action shape: {action.shape}")
	print(f"Sampled reward shape: {reward.shape}")
	
	assert obs.shape[0] == expected_slice_length, \
		f"Expected obs length {expected_slice_length}, got {obs.shape[0]}"
	assert obs.shape[1] == cfg.batch_size, \
		f"Expected batch size {cfg.batch_size}, got {obs.shape[1]}"
	assert action.shape[0] == expected_slice_length - 1
	assert reward.shape[0] == expected_slice_length - 1
	
	print("✓ TrajectoryBuffer tests passed")


if __name__ == "__main__":
	test_trajectory_buffer()
	print("\n✅ All tests passed!")

