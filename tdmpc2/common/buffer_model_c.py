"""
Model C用リプレイバッファ

物理パラメータ + 履歴系列を保存・サンプリング。

通常のBuffer: (obs, action, reward, terminated, task)
Model C用: (obs, action, reward, terminated, task, c_phys, obs_seq, action_seq)
                                                      ^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^
                                                      真の値   GRU用の履歴
"""
import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class ModelCBuffer:
	"""
	Model C用のリプレイバッファ。
	
	特徴:
	- 物理パラメータ c_phys を保存
	- GRU推定用の履歴 (obs_seq, action_seq) を保存
	"""
	
	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0
		
		# 物理パラメータと履歴長
		self.c_phys_dim = getattr(cfg, 'c_phys_dim', 1)
		self.context_length = getattr(cfg, 'context_length', 50)
	
	@property
	def capacity(self):
		"""バッファの容量を返す"""
		return self._capacity
	
	@property
	def num_eps(self):
		"""バッファ内のエピソード数を返す"""
		return self._num_eps
	
	def _reserve_buffer(self, storage):
		"""指定されたストレージでバッファを予約"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)
	
	def _init(self, tds):
		"""リプレイバッファを初期化"""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		
		# バイト数の計算（c_phys と履歴を含む）
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step * self._capacity
		
		print(f'Storage required: {total_bytes/1e9:.2f} GB (including c_phys and history)')
		
		# ヒューリスティック: CUDAまたはCPUメモリを使用するか決定
		storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)
	
	def load(self, td):
		"""バッチでエピソードをロード"""
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		return self._num_eps
	
	def add(self, td):
		"""
		1つのエピソードをバッファに追加。
		
		Args:
			td (TensorDict): エピソードデータ
				- obs, action, reward, terminated, c_phys, obs_history, action_history
		"""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		
		if self._num_eps == 0:
			self._buffer = self._init(td)
		
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps
	
	def _prepare_batch(self, td):
		"""
		サンプリングしたバッチを学習用に前処理。
		
		Returns:
			obs, action, reward, terminated, task, c_phys, obs_seq, action_seq
		"""
		# 基本フィールドを取得
		td = td.select(
			"obs", "action", "reward", "terminated", "task", "c_phys", 
			"obs_history", "action_history",
			strict=False
		).to(self._device, non_blocking=True)
		
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
		
		# 物理パラメータの取得
		c_phys = td.get('c_phys', None)
		if c_phys is not None:
			# c_physはエピソード中一定なので、最初のステップの値を使用
			c_phys = c_phys[0].contiguous()
		else:
			c_phys = torch.zeros(obs.shape[1], self.c_phys_dim, device=self._device)
		
		# 履歴の取得（GRU用）
		obs_history = td.get('obs_history', None)
		action_history = td.get('action_history', None)
		
		if obs_history is not None and action_history is not None:
			# obs_history: (horizon+1, batch, context_length, obs_dim)
			# action_history: (horizon+1, batch, context_length, action_dim)
			# → 最初のステップの履歴を使用
			obs_seq = obs_history[0].contiguous()  # (batch, context_length, obs_dim)
			action_seq = action_history[0].contiguous()  # (batch, context_length, action_dim)
		else:
			# 履歴がない場合はNone（後でゼロで埋める）
			obs_seq = None
			action_seq = None
		
		return obs, action, reward, terminated, task, c_phys, obs_seq, action_seq
	
	def sample(self):
		"""
		バッファから部分軌道のバッチをサンプル。
		
		Returns:
			obs, action, reward, terminated, task, c_phys, obs_seq, action_seq
		"""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return self._prepare_batch(td)

