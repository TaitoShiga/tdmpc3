"""
物理パラメータ推定器（GRU）

履歴エンコーダとして、過去k步の(s, a)系列から
物理コンテキスト c_phys を推定する。

Model Cの「フェーズ1：推定」を担当。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsEstimatorGRU(nn.Module):
	"""
	GRUベースの物理パラメータ推定器。
	
	アーキテクチャ:
	  Input: [(s_t, a_t)] の時系列（過去k步）
	  GRU: 2層、隠れ層256次元
	  Output Head: 線形層 -> c_phys（例：8次元）
	
	学習:
	  L_aux = MSE(c_phys_pred, c_phys_true)
	  GRUはこの損失のみで更新される
	"""
	
	def __init__(
		self,
		obs_dim,
		action_dim,
		c_phys_dim,
		hidden_dim=256,
		num_layers=2,
		dropout=0.1,
		context_length=50,
	):
		"""
		Args:
			obs_dim: 観測の次元
			action_dim: 行動の次元
			c_phys_dim: 物理パラメータの次元
			hidden_dim: GRUの隠れ層次元
			num_layers: GRUの層数
			dropout: ドロップアウト率
			context_length: 履歴長（何ステップ見るか）
		"""
		super().__init__()
		self.obs_dim = obs_dim
		self.action_dim = action_dim
		self.c_phys_dim = c_phys_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.context_length = context_length
		
		# 入力次元: (state, action) を連結
		input_dim = obs_dim + action_dim
		
		# 入力の前処理（オプション：正規化やエンコーディング）
		self.input_proj = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
		)
		
		# GRU本体
		self.gru = nn.GRU(
			input_size=hidden_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			dropout=dropout if num_layers > 1 else 0.0,
		)
		
		# 出力ヘッド: 最終隠れ状態 -> c_phys
		self.output_head = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, c_phys_dim),
			nn.Tanh(),  # 正規化された出力（-1, 1）
		)
	
	def forward(self, obs_seq, action_seq, hidden=None):
		"""
		物理パラメータを推定。
		
		Args:
			obs_seq: 観測系列 (batch, seq_len, obs_dim)
			action_seq: 行動系列 (batch, seq_len, action_dim)
			hidden: GRUの初期隠れ状態（オプション）
		
		Returns:
			c_phys_pred: 推定された物理パラメータ (batch, c_phys_dim)
			hidden: GRUの最終隠れ状態
		"""
		batch_size, seq_len = obs_seq.shape[:2]
		
		# (state, action) を連結
		x = torch.cat([obs_seq, action_seq], dim=-1)  # (batch, seq_len, obs_dim + action_dim)
		
		# 入力の前処理
		x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
		
		# GRUで処理
		output, hidden = self.gru(x, hidden)  # output: (batch, seq_len, hidden_dim)
		
		# 最終ステップの出力を使用
		final_output = output[:, -1, :]  # (batch, hidden_dim)
		
		# 物理パラメータを推定
		c_phys_pred = self.output_head(final_output)  # (batch, c_phys_dim)
		
		return c_phys_pred, hidden
	
	def estimate_from_buffer_data(self, obs_seq, action_seq):
		"""
		バッファからサンプルしたデータから推定（推論モード）。
		
		Args:
			obs_seq: (batch, seq_len, obs_dim)
			action_seq: (batch, seq_len, action_dim)
		
		Returns:
			c_phys_pred: (batch, c_phys_dim)
		"""
		with torch.no_grad():
			c_phys_pred, _ = self.forward(obs_seq, action_seq)
		return c_phys_pred
	
	def compute_loss(self, obs_seq, action_seq, c_phys_true):
		"""
		推定損失 L_aux を計算。
		
		Args:
			obs_seq: (batch, seq_len, obs_dim)
			action_seq: (batch, seq_len, action_dim)
			c_phys_true: 真の物理パラメータ (batch, c_phys_dim)
		
		Returns:
			loss: MSE損失
			info: 統計情報
		"""
		c_phys_pred, _ = self.forward(obs_seq, action_seq)
		
		# MSE損失
		loss = F.mse_loss(c_phys_pred, c_phys_true)
		
		# 統計情報
		with torch.no_grad():
			mae = F.l1_loss(c_phys_pred, c_phys_true)
			max_error = (c_phys_pred - c_phys_true).abs().max()
			
			info = {
				'loss': loss.item(),
				'mae': mae.item(),
				'max_error': max_error.item(),
				'pred_mean': c_phys_pred.mean().item(),
				'pred_std': c_phys_pred.std().item(),
				'true_mean': c_phys_true.mean().item(),
				'true_std': c_phys_true.std().item(),
			}
		
		return loss, info


class PhysicsEstimatorMLP(nn.Module):
	"""
	ベースライン用のMLP推定器（比較用）。
	
	履歴をフラット化してMLPで処理。
	GRUより単純だが、時系列情報を活用しにくい。
	"""
	
	def __init__(
		self,
		obs_dim,
		action_dim,
		c_phys_dim,
		hidden_dim=512,
		num_layers=3,
		dropout=0.1,
		context_length=50,
	):
		super().__init__()
		self.obs_dim = obs_dim
		self.action_dim = action_dim
		self.c_phys_dim = c_phys_dim
		self.context_length = context_length
		
		# 入力次元: (obs + action) * context_length
		input_dim = (obs_dim + action_dim) * context_length
		
		# MLP
		layers = []
		layers.append(nn.Linear(input_dim, hidden_dim))
		layers.append(nn.LayerNorm(hidden_dim))
		layers.append(nn.ReLU())
		layers.append(nn.Dropout(dropout))
		
		for _ in range(num_layers - 1):
			layers.append(nn.Linear(hidden_dim, hidden_dim))
			layers.append(nn.LayerNorm(hidden_dim))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(dropout))
		
		layers.append(nn.Linear(hidden_dim, c_phys_dim))
		layers.append(nn.Tanh())
		
		self.mlp = nn.Sequential(*layers)
	
	def forward(self, obs_seq, action_seq):
		"""
		Args:
			obs_seq: (batch, seq_len, obs_dim)
			action_seq: (batch, seq_len, action_dim)
		
		Returns:
			c_phys_pred: (batch, c_phys_dim)
		"""
		batch_size = obs_seq.shape[0]
		
		# (obs, action)を連結してフラット化
		x = torch.cat([obs_seq, action_seq], dim=-1)  # (batch, seq_len, obs_dim + action_dim)
		x = x.reshape(batch_size, -1)  # (batch, seq_len * (obs_dim + action_dim))
		
		# MLPで処理
		c_phys_pred = self.mlp(x)
		
		return c_phys_pred
	
	def compute_loss(self, obs_seq, action_seq, c_phys_true):
		"""推定損失を計算"""
		c_phys_pred = self.forward(obs_seq, action_seq)
		loss = F.mse_loss(c_phys_pred, c_phys_true)
		
		with torch.no_grad():
			mae = F.l1_loss(c_phys_pred, c_phys_true)
			max_error = (c_phys_pred - c_phys_true).abs().max()
			
			info = {
				'loss': loss.item(),
				'mae': mae.item(),
				'max_error': max_error.item(),
				'pred_mean': c_phys_pred.mean().item(),
				'pred_std': c_phys_pred.std().item(),
			}
		
		return loss, info


def create_physics_estimator(estimator_type, **kwargs):
	"""
	物理推定器のファクトリ関数。
	
	Args:
		estimator_type: 'gru' or 'mlp'
		**kwargs: 推定器のパラメータ
	
	Returns:
		PhysicsEstimator
	"""
	if estimator_type == 'gru':
		return PhysicsEstimatorGRU(**kwargs)
	elif estimator_type == 'mlp':
		return PhysicsEstimatorMLP(**kwargs)
	else:
		raise ValueError(f"Unknown estimator type: {estimator_type}")

