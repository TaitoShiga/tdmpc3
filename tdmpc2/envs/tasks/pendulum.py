import os
import xml.etree.ElementTree as ET

from dm_control.rl import control
from dm_control.suite import pendulum
from dm_control.suite import common
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_DEFAULT_TIME_LIMIT = 20
_TARGET_SPEED = 9.


def get_model_and_assets(mass=1.0):
    """Returns a tuple containing the model XML string and a dict of assets.

    Args:
        mass: Mass value to assign to the distal pendulum sphere.
    """
    model_xml = resources.GetResource(os.path.join(_TASKS_DIR, 'pendulum.xml'))
    if mass != 1.0:
        root = ET.fromstring(model_xml)
        for geom in root.iter('geom'):
            if geom.get('name') == 'mass':
                geom.set('mass', str(mass))
                break
        model_xml = ET.tostring(root, encoding='unicode')
    return model_xml, common.ASSETS


@pendulum.SUITE.add('custom')
def spin(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns pendulum spin task."""
  physics = pendulum.Physics.from_xml_string(*get_model_and_assets())
  task = Spin(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@pendulum.SUITE.add('custom')
def swingup_mass2(time_limit=_DEFAULT_TIME_LIMIT,
                  random=None,
                  environment_kwargs=None):
  """Returns pendulum swing-up task with doubled pole mass."""
  physics = pendulum.Physics.from_xml_string(*get_model_and_assets(mass=2.0))
  task = pendulum.SwingUp(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@pendulum.SUITE.add('custom')
def swingup_mass15(time_limit=_DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
  """Returns pendulum swing-up task with mass=1.5."""
  physics = pendulum.Physics.from_xml_string(*get_model_and_assets(mass=1.5))
  task = pendulum.SwingUp(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@pendulum.SUITE.add('custom')
def swingup_mass25(time_limit=_DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
  """Returns pendulum swing-up task with mass=2.5."""
  physics = pendulum.Physics.from_xml_string(*get_model_and_assets(mass=2.5))
  task = pendulum.SwingUp(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


@pendulum.SUITE.add('custom')
def swingup_mass30(time_limit=_DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
  """Returns pendulum swing-up task with mass=3.0."""
  physics = pendulum.Physics.from_xml_string(*get_model_and_assets(mass=3.0))
  task = pendulum.SwingUp(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class SwingUpRandomized(pendulum.SwingUp):
  """Domain Randomization版Pendulum-Swingup Task
  
  エピソードごとに物理パラメータをランダム化:
  - mass: uniform(0.5, 2.5)
  """
  
  def __init__(self, random=None):
    super().__init__(random=random)
    self._mass_range = (0.5, 2.5)
  
  def initialize_episode(self, physics):
    """エピソードごとに質量をランダム化"""
    # 新しい質量をサンプリング
    mass = self.random.uniform(*self._mass_range)
    
    # Physics内部のモデルを直接変更
    # pendulumの場合、最後のbodyが振り子の質量
    physics.model.body_mass[-1] = mass
    
    # 親クラスの初期化を呼ぶ（初期姿勢のランダム化など）
    super().initialize_episode(physics)


@pendulum.SUITE.add('custom')
def swingup_randomized(time_limit=_DEFAULT_TIME_LIMIT,
                       random=None,
                       environment_kwargs=None):
  """Domain Randomization版Pendulum-Swingup
  
  エピソードごとに物理パラメータをランダム化:
  - mass: uniform(0.5, 2.5)
  
  Transformerが In-Context Learning により多様な物理法則を学習するための環境。
  """
  # デフォルトの質量で環境を作成（実際の質量はinitialize_episodeで設定）
  physics = pendulum.Physics.from_xml_string(*get_model_and_assets(mass=1.0))
  task = SwingUpRandomized(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, **environment_kwargs)


class Spin(pendulum.SwingUp):
  """A custom Pendulum Spin task."""

  def __init__(self, random=None):
    super().__init__(random=random)

  def get_reward(self, physics):
    return rewards.tolerance(np.linalg.norm(physics.angular_velocity()),
                             bounds=(_TARGET_SPEED, float('inf')),
                             margin=_TARGET_SPEED/2,
                             value_at_margin=0.5,
                            sigmoid='linear')
