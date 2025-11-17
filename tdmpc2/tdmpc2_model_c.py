"""
Model C: GRU推定器統合版TD-MPC2

【核心的特徴】2フェーズ分離 + 勾配分離

フェーズ1: GRU推定器が物理パラメータを推定
  - 損失: L_aux = MSE(c_phys_pred, c_phys_true)
  - 更新: GRUのみ

フェーズ2: プランナーが推定された物理パラメータを使用
  - 損失: L_TD-MPC2 (consistency, reward, value, ...)
  - 更新: プランナー（dynamics, reward, Q, pi）のみ
  - 重要: c_physはdetach()されている

使用方法:
    python train.py task=pendulum-swingup-randomized use_model_c=true seed=0
"""
import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model_model_c import WorldModelC
from common.layers import api_model_conversion
from tensordict import TensorDict


class TDMPC2ModelC(torch.nn.Module):
	"""
	Model C版TD-MPC2エージェント。
	
	GRU推定器 + 物理パラメータ条件付きプランナーを統合。
	勾配分離により、2つの学習目標を安定して両立。
	"""
	
	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		
		# Model C用WorldModel
		self.model = WorldModelC(cfg).to(self.device)
		
		# 【重要】2つの独立したOptimizer
		# 1. GRU推定器用
		self.gru_optim = torch.optim.Adam(
			self.model._physics_estimator.parameters(),
			lr=getattr(cfg, 'gru_lr', 3e-4),
			weight_decay=getattr(cfg, 'gru_weight_decay', 1e-5),
		)
		
		# 2. プランナー用（dynamics, reward, Q, encoder）
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
		], lr=self.cfg.lr, capturable=True)
		
		# 3. Policy用
		self.pi_optim = torch.optim.Adam(
			self.model._pi.parameters(), 
			lr=self.cfg.lr, 
			eps=1e-5, 
			capturable=True
		)
		
		self.model.eval()
		self.scale = RunningScale(cfg)
		
		# 大きなアクション空間用のヒューリスティック
		self.cfg.iterations += 2*int(cfg.action_dim >= 20)
		
		# Discount factor
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		
		# MPPI用の前回のmean
		self._prev_mean = torch.nn.Buffer(
			torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		)
		
		# 履歴バッファ（GRU推定用）
		self.context_length = getattr(cfg, 'context_length', 50)
		self._obs_history = []
		self._action_history = []
		
		# Compile（オプション）
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")
	
	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val
	
	def _get_discount(self, episode_length):
		"""エピソード長に応じた割引率を返す"""
		frac = episode_length / self.cfg.discount_denom
		return min(max((frac-1)/frac, self.cfg.discount_min), self.cfg.discount_max)
	
	def save(self, fp):
		"""エージェントのstate dictをファイルに保存"""
		torch.save({
			"model": self.model.state_dict(),
			"gru_optim": self.gru_optim.state_dict(),
			"optim": self.optim.state_dict(),
			"pi_optim": self.pi_optim.state_dict(),
		}, fp)
	
	def load(self, fp):
		"""保存されたstate dictをロード"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		
		# Modelのロード
		model_dict = state_dict["model"] if "model" in state_dict else state_dict
		model_dict = api_model_conversion(self.model.state_dict(), model_dict)
		self.model.load_state_dict(model_dict, strict=False)
		
		# Optimizerのロード（オプション）
		if "gru_optim" in state_dict:
			self.gru_optim.load_state_dict(state_dict["gru_optim"])
		if "optim" in state_dict:
			self.optim.load_state_dict(state_dict["optim"])
		if "pi_optim" in state_dict:
			self.pi_optim.load_state_dict(state_dict["pi_optim"])
		
		return
	
	def load_pretrained_gru(self, fp):
		"""
		事前学習済みのGRU推定器をロード。
		
		Args:
			fp: GRU推定器のチェックポイントパス
		"""
		checkpoint = torch.load(fp, map_location=self.device)
		self.model._physics_estimator.load_state_dict(
			checkpoint['estimator_state_dict']
		)
		print(f'Loaded pretrained GRU from: {fp}')
		print(f'  Val MAE: {checkpoint["val_mae"]:.4f}')
	
	def reset_history(self):
		"""エピソード開始時に履歴をリセット"""
		self._obs_history = []
		self._action_history = []
	
	def update_history(self, obs, action):
		"""履歴を更新"""
		self._obs_history.append(obs.cpu().numpy())
		self._action_history.append(action.cpu().numpy())
		
		# context_lengthを超えたら古いものを削除
		if len(self._obs_history) > self.context_length:
			self._obs_history.pop(0)
			self._action_history.pop(0)
	
	def get_history_tensor(self):
		"""
		履歴をTensorに変換。
		
		Returns:
			obs_seq: (1, seq_len, obs_dim)
			action_seq: (1, seq_len, action_dim)
		"""
		import numpy as np
		
		obs_seq = np.array(self._obs_history)
		action_seq = np.array(self._action_history)
		
		obs_seq = torch.from_numpy(obs_seq).float().unsqueeze(0).to(self.device)
		action_seq = torch.from_numpy(action_seq).float().unsqueeze(0).to(self.device)
		
		return obs_seq, action_seq
	
	@torch.no_grad()
	def estimate_physics_online(self):
		"""
		【フェーズ1】オンラインで物理パラメータを推定。
		
		履歴が十分に溜まっていない場合はゼロベクトルを返す。
		"""
		if len(self._obs_history) < self.context_length:
			# 履歴が不十分な場合
			return torch.zeros(1, self.cfg.c_phys_dim, device=self.device)
		
		obs_seq, action_seq = self.get_history_tensor()
		c_phys_pred = self.model.estimate_physics(obs_seq, action_seq)
		
		return c_phys_pred
	
	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		【フェーズ1+2】物理パラメータを推定してアクションを選択。
		
		Args:
			obs: 環境からの観測
			t0: エピソードの最初の観測かどうか
			eval_mode: 評価モード
			task: タスクインデックス
		
		Returns:
			action: 環境で実行するアクション
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		
		if task is not None:
			task = torch.tensor([task], device=self.device)
		
		# エピソード開始時は履歴をリセット
		if t0:
			self.reset_history()
		
		# 【フェーズ1】物理パラメータを推定
		c_phys = self.estimate_physics_online()
		
		if self.cfg.mpc:
			action = self.plan(obs, c_phys, t0=t0, eval_mode=eval_mode, task=task).cpu()
		else:
			# MPCを使わない場合（ポリシーのみ）
			z = self.model.encode(obs, task)
			action, info = self.model.pi(z, task, c_phys)
			if eval_mode:
				action = info["mean"]
			action = action[0].cpu()
		
		return action
	
	@torch.no_grad()
	def _estimate_value(self, z, actions, task, c_phys):
		"""潜在状態zから始まる軌道の価値を推定"""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task, c_phys), self.cfg)
			z = self.model.next(z, actions[t], task, c_phys)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(
					termination + (self.model.termination(z, task, c_phys) > 0.5).float(), 
					max=1.
				)
		
		action, _ = self.model.pi(z, task, c_phys)
		return G + discount * (1-termination) * self.model.Q(z, action, task, c_phys, return_type='avg')
	
	@torch.no_grad()
	def _plan(self, obs, c_phys, t0=False, eval_mode=False, task=None):
		"""
		【フェーズ2】学習したWorld modelを使ってアクション系列をプラン。
		"""
		# ポリシー軌道のサンプル
		z = self.model.encode(obs, task)
		
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(
				self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, 
				device=self.device
			)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			_c_phys = c_phys.repeat(self.cfg.num_pi_trajs, 1)
			
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task, _c_phys)
				_z = self.model.next(_z, pi_actions[t], task, _c_phys)
			pi_actions[-1], _ = self.model.pi(_z, task, _c_phys)
		
		# 状態とパラメータの初期化
		z = z.repeat(self.cfg.num_samples, 1)
		c_phys = c_phys.repeat(self.cfg.num_samples, 1)
		
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full(
			(self.cfg.horizon, self.cfg.action_dim), 
			self.cfg.max_std, 
			dtype=torch.float, 
			device=self.device
		)
		
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		
		actions = torch.empty(
			self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, 
			device=self.device
		)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions
		
		# MPPI反復
		for _ in range(self.cfg.iterations):
			# アクションのサンプル
			r = torch.randn(
				self.cfg.horizon, 
				self.cfg.num_samples - self.cfg.num_pi_trajs, 
				self.cfg.action_dim, 
				device=std.device
			)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]
			
			# エリートアクションの計算
			value = self._estimate_value(z, actions, task, c_phys).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
			
			# パラメータの更新
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]
		
		# アクションの選択
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)
	
	def update_pi(self, zs, task, c_phys):
		"""潜在状態の系列を使ってポリシーを更新"""
		action, info = self.model.pi(zs, task, c_phys)
		qs = self.model.Q(zs, action, task, c_phys, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)
		
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)
		
		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info
	
	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task, c_phys):
		"""TD-targetを計算"""
		action, _ = self.model.pi(next_z, task, c_phys)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(
			next_z, action, task, c_phys, return_type='min', target=True
		)
	
	def _update(self, obs, action, reward, terminated, task=None, c_phys_true=None, obs_seq=None, action_seq=None):
		"""
		【核心的実装】勾配分離による2フェーズ学習。
		
		Args:
			obs: 観測系列 (horizon+1, batch, obs_dim)
			action: 行動系列 (horizon, batch, action_dim)
			reward: 報酬系列 (horizon, batch, 1)
			terminated: 終了フラグ (horizon, batch, 1)
			task: タスクインデックス
			c_phys_true: 真の物理パラメータ (batch, c_phys_dim)
			obs_seq: GRU用の観測系列 (batch, context_length, obs_dim)
			action_seq: GRU用の行動系列 (batch, context_length, action_dim)
		
		Returns:
			TensorDict: 学習統計情報
		"""
		
		# ========================================
		# フェーズ1: GRU推定器の更新（L_aux）
		# ========================================
		if obs_seq is not None and action_seq is not None and c_phys_true is not None:
			# GRUをtrainモードに設定
			self.model._physics_estimator.train()
			
			loss_aux, info_aux = self.model.compute_physics_estimation_loss(
				obs_seq, action_seq, c_phys_true
			)
			
			# GRU推定器のみ更新
			self.gru_optim.zero_grad(set_to_none=True)
			loss_aux.backward()
			gru_grad_norm = torch.nn.utils.clip_grad_norm_(
				self.model._physics_estimator.parameters(), 
				self.cfg.grad_clip_norm
			)
			self.gru_optim.step()
		else:
			loss_aux = torch.tensor(0.0, device=self.device)
			info_aux = {'mae': 0.0, 'max_error': 0.0}
			gru_grad_norm = torch.tensor(0.0, device=self.device)
		
		# ========================================
		# フェーズ2: プランナーの更新（L_TD-MPC2）
		# ========================================
		
		# 🔑 重要: GRUで推定したc_physをdetach()
		with torch.no_grad():
			if obs_seq is not None and action_seq is not None:
				c_phys_pred = self.model.estimate_physics(obs_seq, action_seq)
			else:
				# 履歴が不十分な場合はゼロベクトル
				c_phys_pred = torch.zeros(obs.shape[1], self.cfg.c_phys_dim, device=self.device)
		
		c_phys = c_phys_pred.detach()  # ← 勾配を切る！
		
		# Targetの計算
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, terminated, task, c_phys)
		
		# 更新の準備
		self.model.train()
		
		# 潜在空間でのロールアウト
		zs = torch.empty(
			self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, 
			device=self.device
		)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task, c_phys)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
			zs[t+1] = z
		
		# 予測
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, c_phys, return_type='all')
		reward_preds = self.model.reward(_zs, action, task, c_phys)
		
		if self.cfg.episodic:
			termination_pred = self.model.termination(zs[1:], task, c_phys, unnormalized=True)
		
		# 損失の計算
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(
			zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))
		):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t
		
		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.
		
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss
		)
		
		# プランナーの更新
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(
			list(self.model._encoder.parameters()) +
			list(self.model._dynamics.parameters()) +
			list(self.model._reward.parameters()) +
			list(self.model._Qs.parameters()) +
			(list(self.model._termination.parameters()) if self.cfg.episodic else []),
			self.cfg.grad_clip_norm
		)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)
		
		# ポリシーの更新
		pi_info = self.update_pi(zs.detach(), task, c_phys)
		
		# Target Q-functionsの更新
		self.model.soft_update_target_Q()
		
		# 学習統計を返す
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"grad_norm": grad_norm,
			"gru_loss_aux": loss_aux,
			"gru_mae": info_aux['mae'],
			"gru_grad_norm": gru_grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()
	
	def update(self, buffer):
		"""
		メインの更新関数。
		
		Args:
			buffer: リプレイバッファ（c_phys_trueとhistoryを含む）
		
		Returns:
			dict: 学習統計情報
		"""
		# バッファからサンプル
		obs, action, reward, terminated, task, c_phys_true, obs_seq, action_seq = buffer.sample()
		
		torch.compiler.cudagraph_mark_step_begin()
		return self._update(
			obs, action, reward, terminated, 
			task=task, 
			c_phys_true=c_phys_true,
			obs_seq=obs_seq,
			action_seq=action_seq
		)

