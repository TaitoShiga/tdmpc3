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
