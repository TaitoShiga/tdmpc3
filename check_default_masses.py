"""各タスクのデフォルト質量を確認"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tdmpc2'))

from dm_control import suite
from envs.tasks import ball_in_cup, reacher, hopper, pendulum

suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)

def check_default_mass(domain, task, body_indices, body_names):
    """デフォルト質量を確認"""
    print(f"\n{'='*60}")
    print(f"{domain}-{task}")
    print(f"{'='*60}")
    
    env = suite.load(domain, task, task_kwargs={'random': 42})
    env.reset()
    
    if isinstance(body_indices, int):
        body_indices = [body_indices]
        body_names = [body_names]
    
    for idx, name in zip(body_indices, body_names):
        mass = env.physics.model.body_mass[idx]
        print(f"  {name} (body[{idx}]): mass = {mass:.6f}")
        
        # 推奨範囲を計算（0.5×～2.5×）
        min_mass = mass * 0.5
        max_mass = mass * 2.5
        print(f"    → 推奨DR範囲 (0.5×～2.5×): ({min_mass:.6f}, {max_mass:.6f})")

print("="*60)
print("デフォルト質量確認 & DR範囲推奨")
print("="*60)

# 1. Pendulum
check_default_mass('pendulum', 'swingup', -1, 'pole')

# 2. Ball-in-Cup
check_default_mass('ball_in_cup', 'catch', 2, 'ball')

# 3. Reacher
check_default_mass('reacher', 'three_easy', [1, 2, 3], ['arm0', 'arm1', 'hand'])

# 4. Hopper
check_default_mass('hopper', 'stand', 1, 'torso')

print(f"\n{'='*60}")
print("推奨: 全タスクでデフォルトの0.5×～2.5×に統一")
print("="*60)

