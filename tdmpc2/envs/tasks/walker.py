import os

from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import walker
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_YOGA_STAND_HEIGHT = 1.0
_YOGA_LIE_DOWN_HEIGHT = 0.08
_YOGA_LEGS_UP_HEIGHT = 1.1

# Default torso mass (estimated from walker.xml, will be verified)
_DEFAULT_TORSO_MASS = 3.34


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return resources.GetResource(os.path.join(_TASKS_DIR, 'walker.xml')), common.ASSETS


@walker.SUITE.add('custom')
def walk_backwards(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Walk Backwards task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = BackwardsPlanarWalker(move_speed=walker._WALK_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def run_backwards(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Run Backwards task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = BackwardsPlanarWalker(move_speed=walker._RUN_SPEED, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def arabesque(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Arabesque task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='arabesque', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def lie_down(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Lie Down task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='lie_down', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def legs_up(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Legs Up task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='legs_up', random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def headstand(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Headstand task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=0, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def flip(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Flip task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=walker._RUN_SPEED*0.75, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


@walker.SUITE.add('custom')
def backflip(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Backflip task."""
  physics = walker.Physics.from_xml_string(*get_model_and_assets())
  task = YogaPlanarWalker(goal='flip', move_speed=-walker._RUN_SPEED*0.75, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
      **environment_kwargs)


# ===== Domain Randomization版Walker Walk =====

class WalkRandomized(walker.PlanarWalker):
    """Domain Randomization版Walker Walk Task
    
    エピソードごとに胴体質量をランダム化:
    - torso_mass: uniform(0.5, 2.5) × default
    
    荷物運搬・ペイロード変化シナリオを想定。
    """
    
    def __init__(self, move_speed=walker._WALK_SPEED, random=None):
        super().__init__(move_speed, random)
        self._torso_mass_range = (0.5 * _DEFAULT_TORSO_MASS, 2.5 * _DEFAULT_TORSO_MASS)
        self.current_torso_mass = _DEFAULT_TORSO_MASS  # デフォルト
    
    def initialize_episode(self, physics):
        """エピソードごとに胴体質量をランダム化"""
        # 新しい胴体質量をサンプリング
        torso_mass = self.random.uniform(*self._torso_mass_range)
        self.current_torso_mass = torso_mass
        
        # Physics内部のモデルを直接変更
        torso_body_id = physics.model.name2id('torso', 'body')
        physics.model.body_mass[torso_body_id] = torso_mass
        
        # 親クラスの初期化を呼ぶ
        super().initialize_episode(physics)


@walker.SUITE.add('custom')
def walk_randomized(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Domain Randomization版Walker Walk
    
    エピソードごとに胴体質量をランダム化:
    - torso_mass: uniform(0.5×, 2.5×) × default
    
    Model CがIn-Context Learningにより質量を学習するための環境。
    """
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = WalkRandomized(move_speed=walker._WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
        **environment_kwargs)


# ===== 固定質量版Walker Walk（Zero-shot評価用） =====

class WalkFixedMass(walker.PlanarWalker):
    """固定胴体質量版Walker Walk Task"""
    
    def __init__(self, torso_mass, move_speed=walker._WALK_SPEED, random=None):
        super().__init__(move_speed, random)
        self.current_torso_mass = torso_mass
    
    def initialize_episode(self, physics):
        """固定胴体質量を設定"""
        torso_body_id = physics.model.name2id('torso', 'body')
        physics.model.body_mass[torso_body_id] = self.current_torso_mass
        super().initialize_episode(physics)


@walker.SUITE.add('custom')
def walk_torso_mass_05x(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Walker Walk with torso_mass=0.5× default"""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = WalkFixedMass(torso_mass=0.5 * _DEFAULT_TORSO_MASS, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
        **environment_kwargs)


@walker.SUITE.add('custom')
def walk_torso_mass_10x(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Walker Walk with torso_mass=1.0× default (baseline)"""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = WalkFixedMass(torso_mass=1.0 * _DEFAULT_TORSO_MASS, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
        **environment_kwargs)


@walker.SUITE.add('custom')
def walk_torso_mass_15x(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Walker Walk with torso_mass=1.5× default"""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = WalkFixedMass(torso_mass=1.5 * _DEFAULT_TORSO_MASS, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
        **environment_kwargs)


@walker.SUITE.add('custom')
def walk_torso_mass_20x(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Walker Walk with torso_mass=2.0× default"""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = WalkFixedMass(torso_mass=2.0 * _DEFAULT_TORSO_MASS, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
        **environment_kwargs)


@walker.SUITE.add('custom')
def walk_torso_mass_25x(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Walker Walk with torso_mass=2.5× default"""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = WalkFixedMass(torso_mass=2.5 * _DEFAULT_TORSO_MASS, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
        **environment_kwargs)


# ===== OOD評価用タスク =====

@walker.SUITE.add('custom')
def walk_torso_mass_03x(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Walker Walk with torso_mass=0.3× default (OOD - light)"""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = WalkFixedMass(torso_mass=0.3 * _DEFAULT_TORSO_MASS, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
        **environment_kwargs)


@walker.SUITE.add('custom')
def walk_torso_mass_30x(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Walker Walk with torso_mass=3.0× default (OOD - heavy)"""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = WalkFixedMass(torso_mass=3.0 * _DEFAULT_TORSO_MASS, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
        **environment_kwargs)


@walker.SUITE.add('custom')
def walk_torso_mass_35x(time_limit=walker._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Walker Walk with torso_mass=3.5× default (OOD - extreme)"""
    physics = walker.Physics.from_xml_string(*get_model_and_assets())
    task = WalkFixedMass(torso_mass=3.5 * _DEFAULT_TORSO_MASS, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=walker._CONTROL_TIMESTEP,
        **environment_kwargs)


class BackwardsPlanarWalker(walker.PlanarWalker):
    """Backwards PlanarWalker task."""
    def __init__(self, move_speed, random=None):
        super().__init__(move_speed, random)
    
    def get_reward(self, physics):
        standing = rewards.tolerance(physics.torso_height(),
                                 bounds=(walker._STAND_HEIGHT, float('inf')),
                                 margin=walker._STAND_HEIGHT/2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3*standing + upright) / 4
        if self._move_speed == 0:
            return stand_reward
        else:
            move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                            bounds=(-float('inf'), -self._move_speed),
                                            margin=self._move_speed/2,
                                            value_at_margin=0.5,
                                            sigmoid='linear')
            return stand_reward * (5*move_reward + 1) / 6


class YogaPlanarWalker(walker.PlanarWalker):
    """Yoga PlanarWalker tasks."""
    
    def __init__(self, goal='arabesque', move_speed=0, random=None):
        super().__init__(0, random)
        self._goal = goal
        self._move_speed = move_speed
    
    def _arabesque_reward(self, physics):
        standing = rewards.tolerance(physics.torso_height(),
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        left_foot_height = physics.named.data.xpos['left_foot', 'z']
        right_foot_height = physics.named.data.xpos['right_foot', 'z']
        left_foot_down = rewards.tolerance(left_foot_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_STAND_HEIGHT/2)
        right_foot_up = rewards.tolerance(right_foot_height,
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        arabesque_reward = (3*standing + left_foot_down + right_foot_up + upright) / 6
        return arabesque_reward
    
    def _lie_down_reward(self, physics):
        torso_down = rewards.tolerance(physics.torso_height(),
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_down = rewards.tolerance(thigh_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        feet_down = rewards.tolerance(feet_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        lie_down_reward = (3*torso_down + thigh_down + upright) / 5
        return lie_down_reward
    
    def _legs_up_reward(self, physics):
        torso_down = rewards.tolerance(physics.torso_height(),
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_down = rewards.tolerance(thigh_height,
                                bounds=(-float('inf'), _YOGA_LIE_DOWN_HEIGHT),
                                margin=_YOGA_LIE_DOWN_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        legs_up = rewards.tolerance(feet_height,
                                bounds=(_YOGA_LEGS_UP_HEIGHT, float('inf')),
                                margin=_YOGA_LEGS_UP_HEIGHT/2)
        upright = (1 - physics.torso_upright()) / 2
        legs_up_reward = (3*torso_down + 2*legs_up + thigh_down + upright) / 7
        return legs_up_reward
    
    def _flip_reward(self, physics):
        thigh_height = (physics.named.data.xpos['left_thigh', 'z'] + physics.named.data.xpos['right_thigh', 'z']) / 2
        thigh_up = rewards.tolerance(thigh_height,
                                bounds=(_YOGA_STAND_HEIGHT, float('inf')),
                                margin=_YOGA_STAND_HEIGHT/2)
        feet_height = (physics.named.data.xpos['left_foot', 'z'] + physics.named.data.xpos['right_foot', 'z']) / 2
        legs_up = rewards.tolerance(feet_height,
                                bounds=(_YOGA_LEGS_UP_HEIGHT, float('inf')),
                                margin=_YOGA_LEGS_UP_HEIGHT/2)
        upside_down_reward = (3*legs_up + 2*thigh_up) / 5
        if self._move_speed == 0:
            return upside_down_reward
        move_reward = rewards.tolerance(physics.horizontal_velocity(),
                                    bounds=(self._move_speed, float('inf')) if self._move_speed > 0 else (-float('inf'), self._move_speed),
                                    margin=abs(self._move_speed)/2,
                                    value_at_margin=0.5,
                                    sigmoid='linear')
        return upside_down_reward * (5*move_reward + 1) / 6
    
    def get_reward(self, physics):
        if self._goal == 'arabesque':
            return self._arabesque_reward(physics)
        elif self._goal == 'lie_down':
            return self._lie_down_reward(physics)
        elif self._goal == 'legs_up':
            return self._legs_up_reward(physics)
        elif self._goal == 'flip':
            return self._flip_reward(physics)
        else:
            raise NotImplementedError(f'Goal {self._goal} is not implemented.')


if __name__ == '__main__':
    env = legs_up()
    obs = env.reset()
    import numpy as np
    next_obs, reward, done, info = env.step(np.zeros(6))
