#!/usr/bin/env python3
"""Domain Randomization タスク連続学習スクリプト（Python版）"""

import subprocess
import sys
import os
from datetime import datetime

# タスク設定
TASKS = [
    {
        'task': 'pendulum-swingup-randomized',
        'exp_name': 'pendulum_dr',
        'steps': 100000,
    },
    {
        'task': 'cup-catch-randomized',
        'exp_name': 'ball_in_cup_dr',
        'steps': 100000,
    },
    {
        'task': 'reacher-three-easy-randomized',
        'exp_name': 'reacher_dr',
        'steps': 100000,
    },
    {
        'task': 'hopper-stand-randomized',
        'exp_name': 'hopper_dr',
        'steps': 100000,
    },
]

# 共通パラメータ
SEED = 0
SAVE_VIDEO = True

def train_task(task_config):
    """単一タスクの学習"""
    task = task_config['task']
    exp_name = task_config['exp_name']
    steps = task_config['steps']
    
    print("\n" + "="*60)
    print(f"Training: {task}")
    print(f"Exp name: {exp_name}")
    print(f"Steps: {steps}")
    print("="*60)
    
    # コマンド構築
    cmd = [
        sys.executable,  # python
        'tdmpc2/train.py',
        f'task={task}',
        f'exp_name={exp_name}',
        f'steps={steps}',
        f'seed={SEED}',
        f'save_video={str(SAVE_VIDEO).lower()}',
        'enable_wandb=false',
        'compile=false',
    ]
    
    # 実行
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Completed: {task}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {task}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted: {task}")
        raise

def main():
    """メイン処理"""
    start_time = datetime.now()
    
    print("="*60)
    print("Domain Randomization タスク連続学習開始")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # ベースディレクトリに移動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    
    results = {}
    
    # 各タスクを連続学習
    for task_config in TASKS:
        try:
            success = train_task(task_config)
            results[task_config['task']] = success
        except KeyboardInterrupt:
            print("\n\n⚠️  ユーザーによって中断されました")
            break
    
    # 結果サマリ
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("学習結果サマリ")
    print("="*60)
    
    for task, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {status}: {task}")
    
    print(f"\n開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"所要時間: {duration}")
    
    # 学習済みモデルの場所を表示
    if any(results.values()):
        print("\n学習済みモデル:")
        for task_config in TASKS:
            if results.get(task_config['task'], False):
                task = task_config['task']
                exp_name = task_config['exp_name']
                print(f"  - logs/{task}/{SEED}/{exp_name}/")
    
    print("\n" + "="*60)
    
    # 全タスク成功なら0、失敗があれば1を返す
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())

