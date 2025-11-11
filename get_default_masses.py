"""デフォルト質量を実測"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tdmpc2'))

from dm_control import suite
from envs.tasks import ball_in_cup, reacher, hopper, pendulum

suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)

# 1. Pendulum
env = suite.load('pendulum', 'swingup', task_kwargs={'random': 42})
env.reset()
pendulum_mass = env.physics.model.body_mass[-1]
print(f"Pendulum pole mass: {pendulum_mass:.6f}")
print(f"  → Current DR: (0.5, 2.5)")
print(f"  → Ratio: {0.5/pendulum_mass:.2f}× ~ {2.5/pendulum_mass:.2f}×")

# 2. Ball-in-Cup
env = suite.load('ball_in_cup', 'catch', task_kwargs={'random': 42})
env.reset()
ball_mass = env.physics.model.body_mass[2]
print(f"\nBall-in-Cup ball mass: {ball_mass:.6f}")
print(f"  → Current DR: (0.003, 0.015)")
print(f"  → Recommended (0.5×~2.5×): ({ball_mass*0.5:.6f}, {ball_mass*2.5:.6f})")

# 3. Reacher
env = suite.load('reacher', 'three_easy', task_kwargs={'random': 42})
env.reset()
reacher_masses = [env.physics.model.body_mass[i] for i in range(1, 4)]
avg_mass = sum(reacher_masses) / len(reacher_masses)
print(f"\nReacher link masses: {[f'{m:.6f}' for m in reacher_masses]}")
print(f"  → Average: {avg_mass:.6f}")
print(f"  → Current DR per link: (0.01, 0.05)")
print(f"  → Recommended (0.5×~2.5× of avg): ({avg_mass*0.5:.6f}, {avg_mass*2.5:.6f})")

# 4. Hopper
env = suite.load('hopper', 'stand', task_kwargs={'random': 42})
env.reset()
hopper_mass = env.physics.model.body_mass[1]
print(f"\nHopper torso mass: {hopper_mass:.6f}")
print(f"  → Current DR: (2.0, 6.0)")
print(f"  → Recommended (0.5×~2.5×): ({hopper_mass*0.5:.6f}, {hopper_mass*2.5:.6f})")

