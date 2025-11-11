"""タスクの物理パラメータを調査するツール

Usage:
    python inspect_task.py cartpole balance
    python inspect_task.py hopper stand
    python inspect_task.py --all  # 全タスクを表示
"""
import sys
import os
import argparse

# tdmpc2ディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tdmpc2'))

from dm_control import suite
from envs.tasks import ball_in_cup, reacher, hopper, pendulum, cheetah, walker, fish

# カスタムタスクを登録
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)


def inspect_task(domain, task, show_all=False):
    """タスクの全パラメータを表示
    
    Args:
        domain: タスクのドメイン名（例: 'cartpole', 'pendulum'）
        task: タスク名（例: 'balance', 'swingup'）
        show_all: Trueの場合、名前なしの要素も表示
    """
    try:
        env = suite.load(domain, task, task_kwargs={'random': 42})
        env.reset()
        physics = env.physics
    except Exception as e:
        print(f"Error loading {domain}-{task}: {e}")
        return
    
    print(f"\n{'='*70}")
    print(f"Task: {domain}-{task}")
    print(f"{'='*70}\n")
    
    # Body masses
    print("=" * 70)
    print("BODIES (mass)")
    print("=" * 70)
    body_names = physics.named.model.body_mass.axes.row.names
    for i, (name, mass) in enumerate(zip(body_names, physics.model.body_mass)):
        if name or show_all:
            display_name = name if name else f"<unnamed_{i}>"
            dr_min, dr_max = mass * 0.5, mass * 2.5
            print(f"  [{i:2d}] {display_name:25s} mass={mass:8.5f}  "
                  f"DR範囲: ({dr_min:.5f}, {dr_max:.5f})")
    
    # Geoms
    print(f"\n{'='*70}")
    print("GEOMS (size, type, friction)")
    print("=" * 70)
    geom_names = physics.named.model.geom_size.axes.row.names
    for i, name in enumerate(geom_names):
        if name or show_all:
            display_name = name if name else f"<unnamed_{i}>"
            size = physics.model.geom_size[i]
            geom_type = physics.model.geom_type[i]
            friction = physics.model.geom_friction[i]
            
            # MuJoCo geom type names
            type_names = ['plane', 'hplane', 'sphere', 'capsule', 'ellipsoid', 
                         'cylinder', 'box', 'mesh']
            type_name = type_names[geom_type] if geom_type < len(type_names) else str(geom_type)
            
            print(f"  [{i:2d}] {display_name:25s} type={type_name:8s} "
                  f"size={size}  friction={friction}")
    
    # Joints
    print(f"\n{'='*70}")
    print("JOINTS (range, damping, stiffness)")
    print("=" * 70)
    jnt_names = physics.named.model.jnt_range.axes.row.names
    for i, name in enumerate(jnt_names):
        if name or show_all:
            display_name = name if name else f"<unnamed_{i}>"
            range_val = physics.model.jnt_range[i]
            damping = physics.model.dof_damping[i]
            
            # stiffness情報も取得
            try:
                stiffness = physics.model.jnt_stiffness[i]
                stiff_str = f"stiffness={stiffness:.4f}"
            except:
                stiff_str = ""
            
            print(f"  [{i:2d}] {display_name:25s} range={range_val}  "
                  f"damping={damping:.4f}  {stiff_str}")
    
    # Actuators
    print(f"\n{'='*70}")
    print("ACTUATORS")
    print("=" * 70)
    try:
        actuator_names = physics.named.model.actuator_ctrlrange.axes.row.names
        for i, name in enumerate(actuator_names):
            if name or show_all:
                display_name = name if name else f"<unnamed_{i}>"
                ctrlrange = physics.model.actuator_ctrlrange[i]
                gear = physics.model.actuator_gear[i]
                print(f"  [{i:2d}] {display_name:25s} ctrlrange={ctrlrange}  gear={gear}")
    except Exception as e:
        print(f"  (No actuators or error: {e})")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Observation dim: {env.observation_spec()['observations'].shape}")
    print(f"  Action dim: {env.action_spec().shape}")
    print(f"  Total bodies: {len(body_names)}")
    print(f"  Total geoms: {len(geom_names)}")
    print(f"  Total joints: {len(jnt_names)}")
    print("=" * 70 + "\n")


def list_available_tasks():
    """利用可能なタスク一覧を表示"""
    print("\n利用可能なタスク:\n")
    
    domains = {}
    for domain, task in suite.ALL_TASKS:
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(task)
    
    for domain in sorted(domains.keys()):
        print(f"  {domain}:")
        for task in sorted(domains[domain]):
            print(f"    - {task}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='タスクの物理パラメータを調査',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python inspect_task.py cartpole balance
  python inspect_task.py hopper stand --show-all
  python inspect_task.py --list
  python inspect_task.py --common  # よく使うタスクを表示
        """
    )
    
    parser.add_argument('domain', nargs='?', help='ドメイン名 (例: cartpole, pendulum)')
    parser.add_argument('task', nargs='?', help='タスク名 (例: balance, swingup)')
    parser.add_argument('--show-all', action='store_true', 
                       help='名前なしの要素も表示')
    parser.add_argument('--list', action='store_true',
                       help='利用可能なタスク一覧を表示')
    parser.add_argument('--common', action='store_true',
                       help='よく使うタスクをまとめて表示')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_tasks()
        return
    
    if args.common:
        # よく使うタスク
        common_tasks = [
            ('cartpole', 'balance'),
            ('cartpole', 'swingup'),
            ('pendulum', 'swingup'),
            ('ball_in_cup', 'catch'),
            ('reacher', 'easy'),
            ('reacher', 'three_easy'),
            ('hopper', 'stand'),
            ('hopper', 'hop'),
            ('cheetah', 'run'),
            ('walker', 'walk'),
        ]
        
        for domain, task in common_tasks:
            inspect_task(domain, task, args.show_all)
        return
    
    if not args.domain or not args.task:
        parser.print_help()
        print("\nヒント: --list で利用可能なタスク一覧を表示")
        return
    
    inspect_task(args.domain, args.task, args.show_all)


if __name__ == '__main__':
    main()

