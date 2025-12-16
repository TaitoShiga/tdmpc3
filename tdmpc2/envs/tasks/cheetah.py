import os
import xml.etree.ElementTree as ET

from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import cheetah
from dm_control.utils import rewards
from dm_control.utils import io as resources
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_CHEETAH_JUMP_HEIGHT = 1.2
_CHEETAH_LIE_HEIGHT = 0.25
_CHEETAH_SPIN_SPEED = 8


def get_model_and_assets(friction=None):
    """Returns a tuple containing the model XML string and a dict of assets.
    
    Args:
        friction: Ground friction value to assign. If None, uses default (0.4).
    """
    model_xml = resources.GetResource(os.path.join(_TASKS_DIR, 'cheetah.xml'))
    if friction is not None:
        root = ET.fromstring(model_xml)
        for geom in root.iter('geom'):
            if geom.get('name') == 'ground':
                # MuJoCoの摩擦は [sliding, torsional, rolling]
                friction_str = f"{friction} {friction*0.25} {friction*0.025}"
                geom.set('friction', friction_str)
                break
        model_xml = ET.tostring(root, encoding='unicode')
    return model_xml, common.ASSETS


@cheetah.SUITE.add('custom')
def run_backwards(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run Backwards task."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = CustomCheetah(goal='run-backwards', move_speed=cheetah._RUN_SPEED*0.8, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def stand_front(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Stand Front task."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = CustomCheetah(goal='stand-front', move_speed=0.5, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def stand_back(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Stand Back task."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = CustomCheetah(goal='stand-back', move_speed=0.5, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def jump(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Jump task."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = CustomCheetah(goal='jump', move_speed=0.5, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def run_front(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run Front task."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = CustomCheetah(goal='run-front', move_speed=cheetah._RUN_SPEED*0.6, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def run_back(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run Back task."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = CustomCheetah(goal='run-back', move_speed=cheetah._RUN_SPEED*0.6, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def lie_down(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Lie Down task."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = CustomCheetah(goal='lie-down', random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def legs_up(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Legs Up task."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = CustomCheetah(goal='legs-up', random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def flip(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Flip task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CustomCheetah(goal='flip', move_speed=cheetah._RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


@cheetah.SUITE.add('custom')
def flip_backwards(time_limit=cheetah._DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Flip Backwards task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = CustomCheetah(goal='flip-backwards', move_speed=cheetah._RUN_SPEED*0.8, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               **environment_kwargs)


class Physics(cheetah.Physics):
    """Physics simulation with additional features for the Cheetah domain."""

    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class CheetahRandomized(cheetah.Cheetah):
    """Domain Randomization版Cheetah Task
    
    エピソードごとに物理パラメータをランダム化:
    - ground_friction: 地面の滑り摩擦係数 uniform(0.2, 0.8)
    """
    
    def __init__(self, random=None):
        super().__init__(random=random)
        self._friction_range = (0.2, 0.8)  # 滑り摩擦の範囲
        self.current_friction = 0.4  # 現在の摩擦係数（デフォルト）
    
    def initialize_episode(self, physics):
        """エピソードごとに地面摩擦をランダム化"""
        # 新しい摩擦係数をサンプリング
        friction = self.random.uniform(*self._friction_range)
        self.current_friction = friction  # 現在の摩擦を記録（Oracleアクセス用）
        
        # Physics内部のモデルを直接変更
        # groundのgeom IDを取得して摩擦を変更
        ground_geom_id = physics.model.name2id('ground', 'geom')
        
        # MuJoCo の friction は [sliding, torsional, rolling] の3要素
        # 主に最初の要素（sliding friction）を変更
        physics.model.geom_friction[ground_geom_id, 0] = friction
        physics.model.geom_friction[ground_geom_id, 1] = friction * 0.25  # torsional
        physics.model.geom_friction[ground_geom_id, 2] = friction * 0.025  # rolling
        
        # 親クラスの初期化を呼ぶ（初期姿勢のランダム化など）
        super().initialize_episode(physics)


@cheetah.SUITE.add('custom')
def run_friction02(time_limit=cheetah._DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
    """Cheetah-Run with low friction (0.2) - slippery surface."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets(friction=0.2))
    task = cheetah.Cheetah(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@cheetah.SUITE.add('custom')
def run_friction04(time_limit=cheetah._DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
    """Cheetah-Run with default friction (0.4)."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets(friction=0.4))
    task = cheetah.Cheetah(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@cheetah.SUITE.add('custom')
def run_friction06(time_limit=cheetah._DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
    """Cheetah-Run with medium-high friction (0.6)."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets(friction=0.6))
    task = cheetah.Cheetah(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@cheetah.SUITE.add('custom')
def run_friction08(time_limit=cheetah._DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
    """Cheetah-Run with high friction (0.8) - grippy surface."""
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets(friction=0.8))
    task = cheetah.Cheetah(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@cheetah.SUITE.add('custom')
def run_randomized(time_limit=cheetah._DEFAULT_TIME_LIMIT,
                   random=None,
                   environment_kwargs=None):
    """Domain Randomization版Cheetah-Run
    
    エピソードごとに物理パラメータをランダム化:
    - ground_friction: uniform(0.2, 0.8)
    
    Transformerが In-Context Learning により多様な物理法則を学習するための環境。
    """
    physics = cheetah.Physics.from_xml_string(*get_model_and_assets())
    task = CheetahRandomized(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class CustomCheetah(cheetah.Cheetah):
    """Custom Cheetah tasks."""
    
    def __init__(self, goal='run-backwards', move_speed=0, random=None):
        super().__init__(random)
        self._goal = goal
        self._move_speed = move_speed

    def _run_backwards_reward(self, physics):
        return rewards.tolerance(physics.speed(),
                            bounds=(-float('inf'), -self._move_speed),
                            margin=self._move_speed,
                            value_at_margin=0,
                            sigmoid='linear')
       
    def _stand_one_foot_reward(self, physics, foot):
        """Note: `foot` is the foot that is *not* on the ground."""
        torso_height = physics.named.data.xpos['torso', 'z']
        foot_height = physics.named.data.xpos[foot, 'z']
        height_reward = rewards.tolerance((torso_height + foot_height)/2,
                            bounds=(_CHEETAH_JUMP_HEIGHT, float('inf')),
                            margin=_CHEETAH_JUMP_HEIGHT/2)
        horizontal_speed_reward = rewards.tolerance(physics.speed(),
                            bounds=(-self._move_speed, self._move_speed),
                            margin=self._move_speed,
                            value_at_margin=0,
                            sigmoid='linear')
        stand_reward = (5*height_reward + horizontal_speed_reward) / 6
        return stand_reward

    def _stand_front_reward(self, physics):
        return self._stand_one_foot_reward(physics, 'bfoot')
    
    def _stand_back_reward(self, physics):
        return self._stand_one_foot_reward(physics, 'ffoot')
    
    def _jump_reward(self, physics):
        front_reward = self._stand_front_reward(physics)
        back_reward = self._stand_back_reward(physics)
        jump_reward = (front_reward + back_reward) / 2
        return jump_reward

    def _run_one_foot_reward(self, physics, foot):
        """Note: `foot` is the foot that is *not* on the ground."""
        torso_height = physics.named.data.xpos['torso', 'z']
        foot_height = physics.named.data.xpos[foot, 'z']
        torso_up = rewards.tolerance(torso_height,
                            bounds=(_CHEETAH_JUMP_HEIGHT, float('inf')),
                            margin=_CHEETAH_JUMP_HEIGHT/2)
        foot_up = rewards.tolerance(foot_height,
                            bounds=(_CHEETAH_JUMP_HEIGHT, float('inf')),
                            margin=_CHEETAH_JUMP_HEIGHT/2)
        up_reward = (3*foot_up + 2*torso_up) / 5
        if self._move_speed == 0:
            return up_reward
        horizontal_speed_reward = rewards.tolerance(physics.speed(),
                            bounds=(self._move_speed, float('inf')),
                            margin=self._move_speed,
                            value_at_margin=0,
                            sigmoid='linear')
        return up_reward * (5*horizontal_speed_reward + 1) / 6

    def _run_front_reward(self, physics):
        return self._run_one_foot_reward(physics, 'bfoot')
    
    def _run_back_reward(self, physics):
        return self._run_one_foot_reward(physics, 'ffoot')

    def _lie_down_reward(self, physics):
        torso_height = physics.named.data.xpos['torso', 'z']
        feet_height = (physics.named.data.xpos['ffoot', 'z'] + physics.named.data.xpos['bfoot', 'z']) / 2
        torso_down = rewards.tolerance(torso_height,
                            bounds=(-float('inf'), _CHEETAH_LIE_HEIGHT),
                            margin=_CHEETAH_LIE_HEIGHT,
                            value_at_margin=0,
                            sigmoid='linear')
        feet_down = rewards.tolerance(feet_height,
                            bounds=(-float('inf'), _CHEETAH_LIE_HEIGHT),
                            margin=_CHEETAH_LIE_HEIGHT,
                            value_at_margin=0,
                            sigmoid='linear')
        lie_down_reward = (3*torso_down + feet_down) / 4
        return lie_down_reward

    def _legs_up_reward(self, physics):
        torso_height = physics.named.data.xpos['torso', 'z']
        torso_down = rewards.tolerance(torso_height,
                            bounds=(-float('inf'), _CHEETAH_LIE_HEIGHT),
                            margin=_CHEETAH_LIE_HEIGHT/2)
        get_up = self._run_one_foot_reward(physics, 'bfoot')
        legs_up_reward = (5*torso_down + get_up) / 6
        return legs_up_reward
    
    def _flip_reward(self, physics, forward=True):
        spin_reward = rewards.tolerance(
                            (1. if forward else -1.) * physics.angmomentum(),
                            bounds=(_CHEETAH_SPIN_SPEED, float('inf')),
                            margin=_CHEETAH_SPIN_SPEED,
                            value_at_margin=0,
                            sigmoid='linear')
        horizontal_speed_reward = rewards.tolerance(
                            (1. if forward else -1.) * physics.speed(),
                            bounds=(self._move_speed, float('inf')),
                            margin=self._move_speed,
                            value_at_margin=0,
                            sigmoid='linear')
        flip_reward = (2*spin_reward + horizontal_speed_reward) / 3
        return flip_reward

    def get_reward(self, physics):
        if self._goal == 'run-backwards':
            return self._run_backwards_reward(physics)
        elif self._goal == 'stand-front':
            return self._stand_front_reward(physics)
        elif self._goal == 'stand-back':
            return self._stand_back_reward(physics)
        elif self._goal == 'jump':
            return self._jump_reward(physics)
        elif self._goal == 'run-front':
            return self._run_front_reward(physics)
        elif self._goal == 'run-back':
            return self._run_back_reward(physics)
        elif self._goal == 'lie-down':
            return self._lie_down_reward(physics)
        elif self._goal == 'legs-up':
            return self._legs_up_reward(physics)
        elif self._goal == 'flip':
            return self._flip_reward(physics, forward=True)
        elif self._goal == 'flip-backwards':
            return self._flip_reward(physics, forward=False)
        else:
            raise NotImplementedError(f'Goal {self._goal} is not implemented.')


if __name__ == '__main__':
    env = jump()
    obs = env.reset()
    import numpy as np
    next_obs, reward, done, info = env.step(np.zeros(6))
    print(reward)
