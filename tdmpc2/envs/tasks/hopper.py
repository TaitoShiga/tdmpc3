import os
import xml.etree.ElementTree as ET

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import hopper
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_CONTROL_TIMESTEP = .02  # (Seconds)

# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 20

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2

# Angular momentum above which reward is 1.
_SPIN_SPEED = 5

# Default thigh length (from hopper.xml line 38)
_DEFAULT_THIGH_LENGTH = 0.33


def get_model_and_assets(thigh_length=None):
	"""Returns a tuple containing the model XML string and a dict of assets.
	
	Args:
		thigh_length: Length of thigh segment (z-axis). If None, uses default (0.33m).
	"""
	model_xml = resources.GetResource(os.path.join(_TASKS_DIR, 'hopper.xml'))
	
	if thigh_length is not None and thigh_length != _DEFAULT_THIGH_LENGTH:
		root = ET.fromstring(model_xml)
		
		# thigh geomの fromto を変更
		for geom in root.iter('geom'):
			if geom.get('name') == 'thigh':
				geom.set('fromto', f'0 0 0 0 0 -{thigh_length}')
				break
		
		# calf bodyの位置を thigh の終端に合わせる
		for body in root.iter('body'):
			if body.get('name') == 'thigh':
				for child_body in body:
					if child_body.tag == 'body' and child_body.get('name') == 'calf':
						child_body.set('pos', f'0 0 -{thigh_length}')
						break
				break
		
		model_xml = ET.tostring(root, encoding='unicode')
	
	return model_xml, common.ASSETS


@hopper.SUITE.add('custom')
def hop_backwards(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Returns the Hop Backwards task."""
	physics = Physics.from_xml_string(*get_model_and_assets())
	task = CustomHopper(goal='hop-backwards', random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


@hopper.SUITE.add('custom')
def hop_backwards_randomized(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Domain Randomization版Hopper Hop Backwards
	
	エピソードごとに大腿長をランダム化:
	- thigh_length: uniform(0.25, 0.45) (デフォルト0.33の ±36%)
	
	構造パラメータの変化に対する適応能力を評価。
	"""
	physics = Physics.from_xml_string(*get_model_and_assets(thigh_length=_DEFAULT_THIGH_LENGTH))
	task = HopBackwardsRandomized(random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


@hopper.SUITE.add('custom')
def flip(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Returns the Flip task."""
	physics = Physics.from_xml_string(*get_model_and_assets())
	task = CustomHopper(goal='flip', random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


@hopper.SUITE.add('custom')
def flip_backwards(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Returns the Flip Backwards task."""
	physics = Physics.from_xml_string(*get_model_and_assets())
	task = CustomHopper(goal='flip-backwards', random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


class Physics(hopper.Physics):

	def angmomentum(self):
		"""Returns the angular momentum of torso of the Cheetah about Y axis."""
		return self.named.data.subtree_angmom['torso'][1]


class CustomHopper(hopper.Hopper):
	"""Custom Hopper tasks."""

	def __init__(self, goal='hop-backwards', random=None):
		super().__init__(None, random)
		self._goal = goal
	
	def _hop_backwards_reward(self, physics):
		standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
		hopping = rewards.tolerance(physics.speed(),
									bounds=(-float('inf'), -_HOP_SPEED/2),
									margin=_HOP_SPEED/4,
									value_at_margin=0.5,
									sigmoid='linear')
		return standing * hopping
	
	def _flip_reward(self, physics, forward=True):
		reward = rewards.tolerance((1. if forward else -1.) * physics.angmomentum(),
								   bounds=(_SPIN_SPEED, float('inf')),
								   margin=_SPIN_SPEED/2,
								   value_at_margin=0,
								   sigmoid='linear')
		return reward


	def get_reward(self, physics):
		if self._goal == 'hop-backwards':
			return self._hop_backwards_reward(physics)
		elif self._goal == 'flip':
			return self._flip_reward(physics, forward=True)
		elif self._goal == 'flip-backwards':
			return self._flip_reward(physics, forward=False)
		else:
			raise NotImplementedError(f'Goal {self._goal} is not implemented.')


class HopBackwardsRandomized(hopper.Hopper):
	"""Domain Randomization版Hopper Hop Backwards Task
	
	エピソードごとに大腿長をランダム化:
	- thigh_length: uniform(0.25, 0.45) (デフォルト0.33の -24%～+36%)
	
	Note: 長さはXML生成時に固定されるため、エピソードごとに
	      XML再生成ではなく、XMLに埋め込んだ値をruntimeで読み出す。
	      
	実装戦略: 現在の thigh_length をタスククラスに記録し、
	         PhysicsParamWrapper から参照できるようにする。
	"""
	
	def __init__(self, random=None):
		super().__init__(None, random)
		self._thigh_length_range = (0.25, 0.45)
		self.current_thigh_length = _DEFAULT_THIGH_LENGTH  # デフォルト
	
	def initialize_episode(self, physics):
		"""エピソードごとに大腿長をランダム化
		
		Note: MuJoCoはgeom fromtoをruntimeで変更できないため、
		      XML再生成が必要。しかしEnvironmentの再構築はコストが高いため、
		      ここでは「XMLに埋め込まれた長さ」を記録するのみ。
		      
		実際の実装では、PhysicsParamWrapperが geom fromto を読み取る。
		"""
		# 新しい大腿長をサンプリング
		thigh_length = self.random.uniform(*self._thigh_length_range)
		self.current_thigh_length = thigh_length
		
		# XML再生成が必要なため、ここでは physics を再構築
		# （これは重い操作だが、エピソード境界なので許容）
		xml_string, assets = get_model_and_assets(thigh_length=thigh_length)
		physics.reload_from_xml_string(xml_string, assets=assets)
		
		# 親クラスの初期化を呼ぶ（初期姿勢のランダム化など）
		super().initialize_episode(physics)
	
	def _hop_backwards_reward(self, physics):
		"""Hop backwards reward (CustomHopperと同じ)"""
		standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
		hopping = rewards.tolerance(physics.speed(),
									bounds=(-float('inf'), -_HOP_SPEED/2),
									margin=_HOP_SPEED/4,
									value_at_margin=0.5,
									sigmoid='linear')
		return standing * hopping
	
	def get_reward(self, physics):
		return self._hop_backwards_reward(physics)


class StandRandomized(hopper.Hopper):
	"""Domain Randomization版Hopper Stand Task
	
	エピソードごとに胴体質量をランダム化:
	- torso_mass: uniform(1.96, 9.8) (デフォルト~3.92の0.5×～2.5×)
	"""
	
	def __init__(self, random=None):
		super().__init__(None, random)
		# デフォルト質量 ~3.92 の 0.5× ~ 2.5×
		self._torso_mass_range = (1.96, 9.8)
	
	def initialize_episode(self, physics):
		"""エピソードごとに胴体質量をランダム化"""
		# 新しい胴体質量をサンプリング
		torso_mass = self.random.uniform(*self._torso_mass_range)
		
		# Physics内部のモデルを直接変更
		# hopperの場合、body index 1がtorso
		physics.model.body_mass[1] = torso_mass
		
		# 親クラスの初期化を呼ぶ
		super().initialize_episode(physics)


@hopper.SUITE.add('custom')
def stand_randomized(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Domain Randomization版Hopper Stand
	
	エピソードごとに胴体質量をランダム化:
	- torso_mass: uniform(2.0, 6.0)
	
	Transformerが In-Context Learning により胴体質量を学習するための環境。
	"""
	physics = hopper.Physics.from_xml_string(*get_model_and_assets())
	task = StandRandomized(random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


class StandFixedMass(hopper.Hopper):
	"""固定胴体質量版Hopper Stand Task"""
	
	def __init__(self, torso_mass, random=None):
		super().__init__(None, random)
		self._torso_mass = torso_mass
	
	def initialize_episode(self, physics):
		"""固定胴体質量を設定"""
		physics.model.body_mass[1] = self._torso_mass
		super().initialize_episode(physics)


# Zero-shot評価用の固定質量版タスク (デフォルト ~3.92)
@hopper.SUITE.add('custom')
def stand_torso_mass_05x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Hopper Stand with torso_mass=1.96 (0.5x)"""
	physics = hopper.Physics.from_xml_string(*get_model_and_assets())
	task = StandFixedMass(torso_mass=1.96, random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


@hopper.SUITE.add('custom')
def stand_torso_mass_10x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Hopper Stand with torso_mass=3.92 (1.0x, baseline)"""
	physics = hopper.Physics.from_xml_string(*get_model_and_assets())
	task = StandFixedMass(torso_mass=3.92, random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


@hopper.SUITE.add('custom')
def stand_torso_mass_15x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Hopper Stand with torso_mass=5.88 (1.5x)"""
	physics = hopper.Physics.from_xml_string(*get_model_and_assets())
	task = StandFixedMass(torso_mass=5.88, random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


@hopper.SUITE.add('custom')
def stand_torso_mass_20x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Hopper Stand with torso_mass=7.84 (2.0x)"""
	physics = hopper.Physics.from_xml_string(*get_model_and_assets())
	task = StandFixedMass(torso_mass=7.84, random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


@hopper.SUITE.add('custom')
def stand_torso_mass_25x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Hopper Stand with torso_mass=9.8 (2.5x)"""
	physics = hopper.Physics.from_xml_string(*get_model_and_assets())
	task = StandFixedMass(torso_mass=9.8, random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


# Zero-shot評価用の固定大腿長版タスク (hop_backwards)
class HopBackwardsFixedLength(hopper.Hopper):
	"""固定大腿長版Hopper Hop Backwards Task"""
	
	def __init__(self, thigh_length, random=None):
		super().__init__(None, random)
		self.current_thigh_length = thigh_length
	
	def _hop_backwards_reward(self, physics):
		standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
		hopping = rewards.tolerance(physics.speed(),
									bounds=(-float('inf'), -_HOP_SPEED/2),
									margin=_HOP_SPEED/4,
									value_at_margin=0.5,
									sigmoid='linear')
		return standing * hopping
	
	def get_reward(self, physics):
		return self._hop_backwards_reward(physics)


@hopper.SUITE.add('custom')
def hop_backwards_length_076x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Hopper Hop Backwards with thigh_length=0.25 (0.76x)"""
	physics = Physics.from_xml_string(*get_model_and_assets(thigh_length=0.25))
	task = HopBackwardsFixedLength(thigh_length=0.25, random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


@hopper.SUITE.add('custom')
def hop_backwards_length_10x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Hopper Hop Backwards with thigh_length=0.33 (1.0x, default)"""
	physics = Physics.from_xml_string(*get_model_and_assets(thigh_length=0.33))
	task = HopBackwardsFixedLength(thigh_length=0.33, random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


@hopper.SUITE.add('custom')
def hop_backwards_length_12x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Hopper Hop Backwards with thigh_length=0.39 (1.18x)"""
	physics = Physics.from_xml_string(*get_model_and_assets(thigh_length=0.39))
	task = HopBackwardsFixedLength(thigh_length=0.39, random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


@hopper.SUITE.add('custom')
def hop_backwards_length_13x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Hopper Hop Backwards with thigh_length=0.43 (1.30x)"""
	physics = Physics.from_xml_string(*get_model_and_assets(thigh_length=0.43))
	task = HopBackwardsFixedLength(thigh_length=0.43, random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


@hopper.SUITE.add('custom')
def hop_backwards_length_136x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
	"""Hopper Hop Backwards with thigh_length=0.45 (1.36x)"""
	physics = Physics.from_xml_string(*get_model_and_assets(thigh_length=0.45))
	task = HopBackwardsFixedLength(thigh_length=0.45, random=random)
	environment_kwargs = environment_kwargs or {}
	return control.Environment(
		physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
		**environment_kwargs)


if __name__ == '__main__':
	env = hop_backwards()
	obs = env.reset()
	import numpy as np
	next_obs, reward, done, info = env.step(np.zeros(2))
	print(reward)
