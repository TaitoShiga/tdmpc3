#!/usr/bin/env python3
"""Domain Randomization タスク評価スクリプト"""

import subprocess
import sys
import os
import glob
from pathlib import Path

# タスク設定
TASKS = [
    {
        'task': 'pendulum-swingup-randomized',
        'exp_name': 'pendulum_dr',
    },
    {
        'task': 'cup-catch-randomized',
        'exp_name': 'ball_in_cup_dr',
    },
    {
        'task': 'reacher-three-easy-randomized',
        'exp_name': 'reacher_dr',
    },
    {
        'task': 'hopper-stand-randomized',
        'exp_name': 'hopper_dr',
    },
]

# 共通パラメータ
SEED = 0
NUM_EPISODES = 10

def find_checkpoint(task, exp_name, seed):
    """最新のチェックポイントを探す"""
    # /tdmpc2/tdmpc2/logs/{task}/{seed}/{exp_name}/models/final.pt
    checkpoint_path = f"/tdmpc2/tdmpc2/logs/{task}/{seed}/{exp_name}/models/final.pt"
    
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    # 見つからない場合、代替パスを探索
    alt_paths = [
        f"/tdmpc2/tdmpc2/logs/{task}/{seed}/{exp_name}/models/*.pt",
        f"tdmpc2/logs/{task}/{seed}/{exp_name}/models/final.pt",
        f"logs/{task}/{seed}/{exp_name}/models/final.pt",
        f"checkpoints/{exp_name}_seed{seed}.pt",
    ]
    
    for pattern in alt_paths:
        matches = glob.glob(pattern)
        if matches:
            return matches[-1]  # 最新のファイル
    
    return None

def evaluate_task(task_config):
    """単一タスクの評価"""
    task = task_config['task']
    exp_name = task_config['exp_name']
    
    print("\n" + "="*60)
    print(f"Evaluating: {task}")
    print(f"Exp name: {exp_name}")
    print("="*60)
    
    # チェックポイントを探す
    checkpoint = find_checkpoint(task, exp_name, SEED)
    
    if checkpoint is None:
        print(f"⚠️  Checkpoint not found for {task}")
        print(f"   Expected: /tdmpc2/tdmpc2/logs/{task}/{SEED}/{exp_name}/models/final.pt")
        return False
    
    print(f"Checkpoint: {checkpoint}")
    
    # コマンド構築
    cmd = [
        sys.executable,  # python
        'tdmpc2/evaluate.py',
        f'task={task}',
        f'checkpoint={checkpoint}',
        f'eval_episodes={NUM_EPISODES}',
        f'save_video=true',
    ]
    
    # 実行
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Completed evaluation: {task}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed evaluation: {task}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted: {task}")
        raise

def main():
    """メイン処理"""
    print("="*60)
    print("Domain Randomization タスク評価開始")
    print("="*60)
    
    # ベースディレクトリに移動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    
    results = {}
    
    # 各タスクを評価
    for task_config in TASKS:
        try:
            success = evaluate_task(task_config)
            results[task_config['task']] = success
        except KeyboardInterrupt:
            print("\n\n⚠️  ユーザーによって中断されました")
            break
    
    # 結果サマリ
    print("\n" + "="*60)
    print("評価結果サマリ")
    print("="*60)
    
    for task, success in results.items():
        status = "✅ Success" if success else "❌ Failed/Not Found"
        print(f"  {status}: {task}")
    
    print("\n" + "="*60)
    
    # 全タスク成功なら0、失敗があれば1を返す
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())

