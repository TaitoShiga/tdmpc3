"""
物理パラメータ取得Wrapper

環境から物理パラメータ（質量、摩擦など）を取得し、
正規化して観測に追加するWrapper。

Model O (Oracle) およびModel C (提案手法) で使用。
"""
import gymnasium as gym
import numpy as np
import torch


class PhysicsParamWrapper(gym.Wrapper):
	"""
	物理パラメータを環境から取得し、正規化するWrapper。
	
	現在サポートしている環境:
	- pendulum-swingup: body_mass[-1] (振り子の質量)
	- ball_in_cup-catch: body_mass[2] (ボールの質量)
	- hopper-stand: body_mass[複数] (Hopperの各部の質量)
	- hopper-hop_backwards: thigh_length (大腿長)
	- cheetah-run: geom_friction['ground'] (地面の摩擦係数)
	- reacher-three_easy: 未実装
	
	Args:
		env: 元の環境
		param_type: 'mass', 'friction', 'damping', 'length' など
		param_indices: 取得する物理パラメータのインデックス (Noneの場合は自動検出)
		normalization: 'standard' (平均=0, 標準偏差=1) または 'minmax' (範囲を[-1, 1]に)
		default_value: デフォルトの物理パラメータ値（正規化の基準）
		scale: 正規化のスケール（標準偏差または範囲の半分）
	"""
	
	def __init__(
		self,
		env,
		param_type='mass',
		param_indices=None,
		normalization='standard',
		default_value=None,
		scale=None,
		domain=None,
		task=None,
	):
		super().__init__(env)
		self.param_type = param_type
		self.param_indices = param_indices
		self.normalization = normalization
		self.domain = domain
		self.task = task
		
		# 環境に応じたデフォルト値の設定
		if default_value is None or scale is None:
			self._set_default_params()
		else:
			self.default_value = default_value
			self.scale = scale
		
		# パラメータ次元の取得
		self.c_phys_dim = self._get_param_dim()
		
		# 現在の物理パラメータ（キャッシュ）
		self._current_c_phys = None
	
	def _set_default_params(self):
		"""環境に応じたデフォルト値とスケールを設定"""
		if self.param_type == 'friction':
			# Friction parameters
			if self.domain == 'cheetah':
				# Cheetah: デフォルト摩擦=0.4, DRの範囲=(0.2, 0.8)
				# 平均=0.5, 標準偏差≈0.173
				self.default_value = np.array([0.4])
				self.scale = np.array([0.3])  # (0.8-0.2)/2 = 0.3
			else:
				# 汎用的なデフォルト
				self.default_value = np.array([0.5])
				self.scale = np.array([0.3])
		
		elif self.param_type == 'mass':
			# Mass parameters
			if self.domain == 'pendulum':
				# Pendulum: デフォルト質量=1.0, DRの範囲=(0.5, 2.5)
				# 平均=1.5, 標準偏差≈0.577
				self.default_value = np.array([1.0])
				self.scale = np.array([1.0])  # 質量1.0を基準とした正規化
			
			elif self.domain == 'ball_in_cup':
				# Ball-in-Cup: デフォルト質量≈0.006, DRの範囲=(0.003, 0.015)
				self.default_value = np.array([0.006])
				self.scale = np.array([0.006])
			
			elif self.domain == 'hopper':
				# Hopper: 複数の質量パラメータ
				# とりあえず主要な3つ（torso, thigh, leg）を使用
				self.default_value = np.array([3.92699082, 3.53429174, 2.71433605])
				self.scale = np.array([3.92699082, 3.53429174, 2.71433605])
			
			elif self.domain == 'reacher':
				# Reacher: 複数の質量パラメータ
				# とりあえずarm0, arm1, fingerを使用
				self.default_value = np.array([0.0, 0.0, 0.01])
				self.scale = np.array([0.01, 0.01, 0.01])
			
			else:
				# 汎用的なデフォルト
				self.default_value = np.array([1.0])
				self.scale = np.array([1.0])
		
		elif self.param_type == 'length':
			# Length parameters (geometric/structural)
			if self.domain == 'hopper':
				# Hopper: 大腿長 (thigh length)
				# デフォルト=0.33, DRの範囲=(0.25, 0.45)
				# 平均=0.35, 標準偏差≈0.058
				self.default_value = np.array([0.33])
				self.scale = np.array([0.1])  # (0.45-0.25)/2 = 0.1
			else:
				# 汎用的なデフォルト
				self.default_value = np.array([1.0])
				self.scale = np.array([0.5])
		
		else:
			# 汎用的なデフォルト
			self.default_value = np.array([1.0])
			self.scale = np.array([1.0])
	
	def _get_param_dim(self):
		"""物理パラメータの次元を取得"""
		if self.param_indices is not None:
			if isinstance(self.param_indices, int):
				return 1
			return len(self.param_indices)
		
		# 環境に応じた自動検出
		if self.param_type == 'friction':
			# Friction: 通常は1次元（地面の摩擦係数）
			if self.domain == 'cheetah':
				return 1
			else:
				return 1
		
		elif self.param_type == 'mass':
			if self.domain == 'pendulum':
				return 1
			elif self.domain == 'ball_in_cup':
				return 1
			elif self.domain == 'hopper':
				return 3  # torso, thigh, leg
			elif self.domain == 'reacher':
				return 3  # arm0, arm1, finger
			else:
				return 1
		
		elif self.param_type == 'length':
			# Length: 通常は1次元（単一リンク長）
			if self.domain == 'hopper':
				return 1  # thigh_length
			else:
				return 1
		
		else:
			return 1
	
	def _get_raw_physics_param(self):
		"""環境から生の物理パラメータを取得"""
		try:
			# DMControl環境の場合
			physics = self.env.unwrapped.physics
			
			if self.param_type == 'mass':
				if self.param_indices is not None:
					if isinstance(self.param_indices, int):
						return np.array([physics.model.body_mass[self.param_indices]])
					return physics.model.body_mass[self.param_indices]
				
				# 環境に応じた自動検出
				if self.domain == 'pendulum':
					# Pendulum: 最後のbodyが振り子の質量
					return np.array([physics.model.body_mass[-1]])
				
				elif self.domain == 'ball_in_cup':
					# Ball-in-Cup: body index 2がボール
					return np.array([physics.model.body_mass[2]])
				
				elif self.domain == 'hopper':
					# Hopper: 主要な3つの質量
					# body_mass: [world, torso, thigh, leg, foot]
					return physics.model.body_mass[1:4]  # torso, thigh, leg
				
				elif self.domain == 'reacher':
					# Reacher: arm0, arm1, finger
					# body_mass: [world, arm0, arm1, finger, target]
					return physics.model.body_mass[1:4]  # arm0, arm1, finger
			
			elif self.param_type == 'friction':
				# 摩擦パラメータの取得
				if self.param_indices is not None:
					if isinstance(self.param_indices, int):
						# 特定のgeomの摩擦を取得（sliding frictionのみ）
						return np.array([physics.model.geom_friction[self.param_indices, 0]])
					# 複数のgeomの摩擦
					return physics.model.geom_friction[self.param_indices, 0]
				
				# 環境に応じた自動検出
				if self.domain == 'cheetah':
					# Cheetah: groundのgeomの摩擦を取得
					try:
						ground_geom_id = physics.model.name2id('ground', 'geom')
						return np.array([physics.model.geom_friction[ground_geom_id, 0]])
					except:
						return self.default_value.copy()
				
				else:
					# その他の環境: 最初のgeomの摩擦
					return np.array([physics.model.geom_friction[0, 0]])
			
			elif self.param_type == 'length':
				# 長さパラメータの取得
				if self.domain == 'hopper':
					# Hopper: thigh geomの長さを取得
					# 方法1: タスククラスから直接取得（推奨）
					try:
						task = self.env.unwrapped.task
						if hasattr(task, 'current_thigh_length'):
							return np.array([task.current_thigh_length])
					except:
						pass
					
					# 方法2: geom fromto から計算（フォールバック）
					try:
						thigh_geom_id = physics.model.name2id('thigh', 'geom')
						# MuJoCoでgeom fromtoを取得するのは難しいため、
						# geom_pos と geom_size から推定
						# ただしcapsuleの長さは fromto の差分なので直接取れない
						# → XMLから読むか、タスククラスに記録しておく必要がある
						
						# とりあえずデフォルト値を返す（実際の実装ではタスククラスから取得）
						return self.default_value.copy()
					except:
						return self.default_value.copy()
				else:
					return self.default_value.copy()
			
			elif self.param_type == 'damping':
				# TODO: ダンピングパラメータの取得
				raise NotImplementedError("Damping parameter extraction not implemented")
		
		except Exception as e:
			print(f"Warning: Failed to extract physics parameter: {e}")
			# フォールバック: デフォルト値を返す
			return self.default_value.copy()
		
		# フォールバック
		return self.default_value.copy()
	
	def _normalize(self, raw_param):
		"""物理パラメータを正規化"""
		if self.normalization == 'standard':
			# (x - μ) / σ の形式
			return (raw_param - self.default_value) / (self.scale + 1e-8)
		
		elif self.normalization == 'minmax':
			# [-1, 1] に正規化
			# 仮定: 範囲が [default - scale, default + scale]
			normalized = (raw_param - self.default_value) / (self.scale + 1e-8)
			return np.clip(normalized, -1.0, 1.0)
		
		elif self.normalization == 'none':
			return raw_param
		
		else:
			raise ValueError(f"Unknown normalization: {self.normalization}")
	
	def get_physics_param(self):
		"""
		正規化された物理パラメータを取得。
		
		Returns:
			torch.Tensor: 正規化された物理パラメータ (c_phys_dim,)
		"""
		raw = self._get_raw_physics_param()
		normalized = self._normalize(raw)
		return torch.from_numpy(normalized).float()
	
	def reset(self, **kwargs):
		"""環境をリセットし、物理パラメータを更新"""
		obs = self.env.reset(**kwargs)
		# リセット後の物理パラメータを取得してキャッシュ
		self._current_c_phys = self.get_physics_param()
		return obs
	
	def step(self, action):
		"""
		1ステップ実行。
		
		Note: 物理パラメータは基本的にエピソード中は変わらないため、
		      stepごとに再取得する必要はないが、念のため更新可能にしておく。
		"""
		obs, reward, done, info = self.env.step(action)
		# infoに物理パラメータを追加（オプション）
		# info['c_phys'] = self._current_c_phys
		return obs, reward, done, info
	
	@property
	def current_c_phys(self):
		"""現在の物理パラメータ（キャッシュ）"""
		if self._current_c_phys is None:
			self._current_c_phys = self.get_physics_param()
		return self._current_c_phys
	
	@current_c_phys.setter
	def current_c_phys(self, value):
		"""キャッシュを書き換えるためのセッター（動画用評価で使用）"""
		if value is None:
			self._current_c_phys = None
			return
		tensor = torch.as_tensor(value, dtype=torch.float32)
		# Oracleのactなどは1次元Tensorを想定するためflattenして保持
		self._current_c_phys = tensor.view(-1)


def wrap_with_physics_param(env, cfg):
	"""
	環境を物理パラメータ取得Wrapperでラップ。
	
	Args:
		env: 元の環境
		cfg: 設定オブジェクト
	
	Returns:
		PhysicsParamWrapper: ラップされた環境
	"""
	# タスク名から domain と task を抽出
	domain, task = cfg.task.replace('-', '_').split('_', 1)
	domain = dict(cup='ball_in_cup').get(domain, domain)
	
	return PhysicsParamWrapper(
		env=env,
		param_type=getattr(cfg, 'phys_param_type', 'mass'),
		param_indices=getattr(cfg, 'phys_param_indices', None),
		normalization=getattr(cfg, 'phys_param_normalization', 'standard'),
		default_value=getattr(cfg, 'phys_param_default', None),
		scale=getattr(cfg, 'phys_param_scale', None),
		domain=domain,
		task=task,
	)
