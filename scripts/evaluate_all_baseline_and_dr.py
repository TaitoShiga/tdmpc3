#!/usr/bin/env python3
"""Baseline + DR タスク評価スクリプト"""

import subprocess
import sys
import os
import glob
from pathlib import Path

# タスク設定（train_all_baseline_and_dr.pyと同じ）
TASK_PAIRS = [
    {
        'baseline': {
            'task': 'cup-catch',
            'exp_name': 'ball_in_cup_baseline',
        },
        'dr': {
            'task': 'cup-catch-randomized',
            'exp_name': 'ball_in_cup_dr',
        }
    },
    {
        'baseline': {
            'task': 'reacher-three-easy',
            'exp_name': 'reacher_baseline',
        },
        'dr': {
            'task': 'reacher-three-easy-randomized',
            'exp_name': 'reacher_dr',
        }
    },
    {
        'baseline': {
            'task': 'hopper-stand',
            'exp_name': 'hopper_baseline',
        },
        'dr': {
            'task': 'hopper-stand-randomized',
            'exp_name': 'hopper_dr',
        }
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

def evaluate_task(task_config, task_type):
    """単一タスクの評価"""
    task = task_config['task']
    exp_name = task_config['exp_name']
    
    print("\n" + "="*60)
    print(f"Evaluating: {task} [{task_type}]")
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
        print(f"✅ Completed evaluation: {task} [{task_type}]")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed evaluation: {task} [{task_type}]")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted: {task} [{task_type}]")
        raise

def main():
    """メイン処理"""
    print("="*60)
    print("Baseline + DR タスク評価開始")
    print("="*60)
    print(f"\n評価するタスク: {len(TASK_PAIRS)} × 2 = {len(TASK_PAIRS)*2} 個\n")
    
    # ベースディレクトリに移動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    
    results = {}
    
    # 各タスクペアを評価
    for idx, task_pair in enumerate(TASK_PAIRS, 1):
        print(f"\n{'#'*60}")
        print(f"タスクペア {idx}/{len(TASK_PAIRS)}")
        print(f"{'#'*60}")
        
        # Baseline版
        try:
            baseline_config = task_pair['baseline']
            success = evaluate_task(baseline_config, 'BASELINE')
            results[f"{baseline_config['task']} [baseline]"] = success
        except KeyboardInterrupt:
            print("\n\n⚠️  ユーザーによって中断されました")
            break
        
        # DR版
        try:
            dr_config = task_pair['dr']
            success = evaluate_task(dr_config, 'DR')
            results[f"{dr_config['task']} [dr]"] = success
        except KeyboardInterrupt:
            print("\n\n⚠️  ユーザーによって中断されました")
            break
    
    # 結果サマリ
    print("\n" + "="*60)
    print("評価結果サマリ")
    print("="*60)
    
    # タスクペアごとに結果表示
    for task_pair in TASK_PAIRS:
        baseline_task = task_pair['baseline']['task']
        dr_task = task_pair['dr']['task']
        
        baseline_key = f"{baseline_task} [baseline]"
        dr_key = f"{dr_task} [dr]"
        
        if baseline_key in results and dr_key in results:
            baseline_status = "✅" if results[baseline_key] else "❌"
            dr_status = "✅" if results[dr_key] else "❌"
            domain = baseline_task.split('-')[0]
            print(f"\n  {domain.upper()}:")
            print(f"    {baseline_status} Baseline: {baseline_task}")
            print(f"    {dr_status} DR:       {dr_task}")
    
    print("\n" + "="*60)
    
    # 全タスク成功なら0、失敗があれば1を返す
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())

