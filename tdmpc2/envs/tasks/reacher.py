import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import reacher
from dm_control.utils import io as resources
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = .05
_SMALL_TARGET = .015


def get_model_and_assets(links):
    """Returns a tuple containing the model XML string and a dict of assets."""
    assert links in {3, 4}, 'Only 3 or 4 links are supported.'
    fn = 'reacher_three_links.xml' if links == 3 else 'reacher_four_links.xml'
    return resources.GetResource(os.path.join(_TASKS_DIR, fn)), common.ASSETS


@reacher.SUITE.add('custom')
def three_easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns three-link reacher with sparse reward with 5e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets(links=3))
  task = CustomThreeLinkReacher(target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def three_hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns three-link reacher with sparse reward with 1e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets(links=3))
  task = CustomThreeLinkReacher(target_size=_SMALL_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns three-link reacher with sparse reward with 5e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = CustomThreeLinkReacher(target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns three-link reacher with sparse reward with 1e-2 tol and randomized target."""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = CustomThreeLinkReacher(target_size=_SMALL_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Reacher domain."""

  def finger_to_target(self):
    """Returns the vector from target to finger in global coordinates."""
    return (self.named.data.geom_xpos['target', :2] -
            self.named.data.geom_xpos['finger', :2])

  def finger_to_target_dist(self):
    """Returns the signed distance between the finger and target surface."""
    return np.linalg.norm(self.finger_to_target())


class CustomThreeLinkReacher(reacher.Reacher):
  """Custom Reacher tasks."""

  def __init__(self, target_size, random=None):
    super().__init__(target_size, random)

  def get_observation(self, physics):
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['to_target'] = physics.finger_to_target()
    obs['velocity'] = physics.velocity()
    return obs


class ThreeEasyRandomized(reacher.Reacher):
  """Domain Randomization版3-Link Reacher Easy Task
  
  エピソードごとにリンク質量をランダム化:
  - link_mass: uniform(0.01, 0.05) for each link (デフォルト~0.02の0.5×～2.5×)
  """
  
  def __init__(self, target_size, random=None):
    super().__init__(target_size, random)
    # デフォルト質量 ~0.02 の 0.5× ~ 2.5×
    self._link_mass_range = (0.01, 0.05)
  
  def initialize_episode(self, physics):
    """エピソードごとにリンク質量をランダム化"""
    # 各リンクの質量をサンプリング
    # 3-link reacherの場合、body 1,2,3がarm0, arm1, hand
    for i in range(1, 4):
      link_mass = self.random.uniform(*self._link_mass_range)
      physics.model.body_mass[i] = link_mass
    
    # 親クラスの初期化を呼ぶ
    super().initialize_episode(physics)


@reacher.SUITE.add('custom')
def three_easy_randomized(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Domain Randomization版3-Link Reacher Easy
  
  エピソードごとにリンク質量をランダム化:
  - link_mass: uniform(0.01, 0.05) for each link
  
  Transformerが In-Context Learning によりリンク質量を学習するための環境。
  """
  physics = Physics.from_xml_string(*get_model_and_assets(links=3))
  task = ThreeEasyRandomized(target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class ThreeEasyFixedMass(reacher.Reacher):
  """固定リンク質量版3-Link Reacher Easy Task"""
  
  def __init__(self, link_mass, target_size, random=None):
    super().__init__(target_size, random)
    self._link_mass = link_mass
  
  def initialize_episode(self, physics):
    """固定リンク質量を設定"""
    for i in range(1, 4):  # body 1,2,3がarm0, arm1, hand
      physics.model.body_mass[i] = self._link_mass
    super().initialize_episode(physics)


# Zero-shot評価用の固定質量版タスク (デフォルト ~0.02)
@reacher.SUITE.add('custom')
def three_easy_link_mass_05x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """3-Link Reacher Easy with link_mass=0.01 (0.5x)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=3))
  task = ThreeEasyFixedMass(link_mass=0.01, target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def three_easy_link_mass_10x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """3-Link Reacher Easy with link_mass=0.02 (1.0x, baseline)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=3))
  task = ThreeEasyFixedMass(link_mass=0.02, target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def three_easy_link_mass_15x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """3-Link Reacher Easy with link_mass=0.03 (1.5x)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=3))
  task = ThreeEasyFixedMass(link_mass=0.03, target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def three_easy_link_mass_20x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """3-Link Reacher Easy with link_mass=0.04 (2.0x)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=3))
  task = ThreeEasyFixedMass(link_mass=0.04, target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def three_easy_link_mass_25x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """3-Link Reacher Easy with link_mass=0.05 (2.5x)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=3))
  task = ThreeEasyFixedMass(link_mass=0.05, target_size=_BIG_TARGET, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


# ===== Reacher-Hard Perturbation Tasks =====
# デフォルト値（4-link reacher）:
#   - actuator_gear: 0.05
#   - joint_damping: 0.01
#   - body_mass: ~0.02 per link
#   - joint_armature: ~0.001 (XMLにない場合のMuJoCoデフォルト)

_DEFAULT_ACTUATOR_GEAR = 0.05
_DEFAULT_JOINT_DAMPING = 0.01
_DEFAULT_LINK_MASS = 0.02
_DEFAULT_ARMATURE = 0.001


# ===== 1. Actuator Gear Perturbations =====

class ReacherHardActuator(reacher.Reacher):
  """Actuator gear摂動版 Reacher-Hard"""
  
  def __init__(self, actuator_scale, target_size=_SMALL_TARGET, random=None):
    super().__init__(target_size, random)
    self._actuator_scale = actuator_scale
    self._base_gear = None
    self.current_actuator_scale = actuator_scale
  
  def initialize_episode(self, physics):
    if self._base_gear is None:
      self._base_gear = physics.model.actuator_gear.copy()
    physics.model.actuator_gear[:] = self._base_gear * self._actuator_scale
    super().initialize_episode(physics)


@reacher.SUITE.add('custom')
def four_hard_actuator_04x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with actuator_gear=0.4x (弱い)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardActuator(actuator_scale=0.4, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_actuator_06x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with actuator_gear=0.6x"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardActuator(actuator_scale=0.6, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_actuator_08x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with actuator_gear=0.8x"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardActuator(actuator_scale=0.8, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_actuator_10x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with actuator_gear=1.0x (baseline)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardActuator(actuator_scale=1.0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_actuator_12x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with actuator_gear=1.2x (強い)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardActuator(actuator_scale=1.2, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_actuator_14x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with actuator_gear=1.4x (強い)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardActuator(actuator_scale=1.4, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


# ===== 2. Joint Damping Perturbations =====

class ReacherHardDamping(reacher.Reacher):
  """Joint damping摂動版 Reacher-Hard"""
  
  def __init__(self, damping_scale, target_size=_SMALL_TARGET, random=None):
    super().__init__(target_size, random)
    self._damping_scale = damping_scale
    self._base_damping = None
    self.current_damping_scale = damping_scale
  
  def initialize_episode(self, physics):
    if self._base_damping is None:
      self._base_damping = physics.model.dof_damping.copy()
    physics.model.dof_damping[:] = self._base_damping * self._damping_scale
    super().initialize_episode(physics)


@reacher.SUITE.add('custom')
def four_hard_damping_05x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with damping=0.5x (低い - 振動しやすい)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardDamping(damping_scale=0.5, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_damping_075x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with damping=0.75x"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardDamping(damping_scale=0.75, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_damping_10x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with damping=1.0x (baseline)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardDamping(damping_scale=1.0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_damping_15x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with damping=1.5x (高い - 動きが鈍い)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardDamping(damping_scale=1.5, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_damping_20x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with damping=2.0x (高い - 動きが鈍い)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardDamping(damping_scale=2.0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


# ===== 3. Link Mass Perturbations =====

class ReacherHardMass(reacher.Reacher):
  """Link mass摂動版 Reacher-Hard"""
  
  def __init__(self, mass_scale, target_size=_SMALL_TARGET, random=None):
    super().__init__(target_size, random)
    self._mass_scale = mass_scale
    self._base_mass = None
    self.current_mass_scale = mass_scale
  
  def initialize_episode(self, physics):
    if self._base_mass is None:
      # 4-link reacherの場合、body 1-5がarm0, arm1, arm2, hand, finger
      self._base_mass = physics.model.body_mass[1:6].copy()
    physics.model.body_mass[1:6] = self._base_mass * self._mass_scale
    super().initialize_episode(physics)


@reacher.SUITE.add('custom')
def four_hard_mass_05x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with link_mass=0.5x (軽い)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardMass(mass_scale=0.5, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_mass_075x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with link_mass=0.75x"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardMass(mass_scale=0.75, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_mass_10x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with link_mass=1.0x (baseline)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardMass(mass_scale=1.0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_mass_15x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with link_mass=1.5x (重い)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardMass(mass_scale=1.5, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_mass_20x(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with link_mass=2.0x (重い)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardMass(mass_scale=2.0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


# ===== 4. Joint Armature Perturbations =====

class ReacherHardArmature(reacher.Reacher):
  """Joint armature摂動版 Reacher-Hard"""
  
  def __init__(self, armature_value, target_size=_SMALL_TARGET, random=None):
    super().__init__(target_size, random)
    self._armature_value = armature_value
    self.current_armature = armature_value
  
  def initialize_episode(self, physics):
    # Armatureを設定（全ての自由度）
    physics.model.dof_armature[:] = self._armature_value
    super().initialize_episode(physics)


@reacher.SUITE.add('custom')
def four_hard_armature_low(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with armature=0.0005 (低い慣性)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardArmature(armature_value=0.0005, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_armature_mid(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with armature=0.001 (baseline)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardArmature(armature_value=0.001, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@reacher.SUITE.add('custom')
def four_hard_armature_high(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Reacher-Hard with armature=0.002 (高い慣性)"""
  physics = Physics.from_xml_string(*get_model_and_assets(links=4))
  task = ReacherHardArmature(armature_value=0.002, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)
