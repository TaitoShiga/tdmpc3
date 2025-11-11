#!/usr/bin/env python3
"""Domain Randomization タスク＋ベースライン連続学習スクリプト

各タスクについて、ベースライン版とDR版の両方を学習する。
"""

import subprocess
import sys
import os
from datetime import datetime

# タスク設定
TASK_PAIRS = [
    {
        'baseline': {
            'task': 'cup-catch',
            'exp_name': 'ball_in_cup_baseline',
            'steps': 100000,
        },
        'dr': {
            'task': 'cup-catch-randomized',
            'exp_name': 'ball_in_cup_dr',
            'steps': 100000,
        }
    },
    {
        'baseline': {
            'task': 'reacher-three-easy',
            'exp_name': 'reacher_baseline',
            'steps': 100000,
        },
        'dr': {
            'task': 'reacher-three-easy-randomized',
            'exp_name': 'reacher_dr',
            'steps': 100000,
        }
    },
    {
        'baseline': {
            'task': 'hopper-stand',
            'exp_name': 'hopper_baseline',
            'steps': 100000,
        },
        'dr': {
            'task': 'hopper-stand-randomized',
            'exp_name': 'hopper_dr',
            'steps': 100000,
        }
    },
]

# 共通パラメータ
SEED = 0
SAVE_VIDEO = True

def train_task(task_config, task_type):
    """単一タスクの学習"""
    task = task_config['task']
    exp_name = task_config['exp_name']
    steps = task_config['steps']
    
    print("\n" + "="*60)
    print(f"Training: {task} [{task_type}]")
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
        print(f"✅ Completed: {task} [{task_type}]")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {task} [{task_type}]")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted: {task} [{task_type}]")
        raise

def main():
    """メイン処理"""
    start_time = datetime.now()
    
    print("="*60)
    print("Baseline + DR タスク連続学習開始")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\n学習するタスク: {len(TASK_PAIRS)} × 2 = {len(TASK_PAIRS)*2} 個")
    print("  - Baseline版（通常の固定パラメータ）")
    print("  - DR版（Domain Randomization）\n")
    
    # ベースディレクトリに移動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    
    results = {}
    
    # 各タスクペアを学習
    for idx, task_pair in enumerate(TASK_PAIRS, 1):
        print(f"\n{'#'*60}")
        print(f"タスクペア {idx}/{len(TASK_PAIRS)}")
        print(f"{'#'*60}")
        
        # Baseline版
        try:
            baseline_config = task_pair['baseline']
            success = train_task(baseline_config, 'BASELINE')
            results[f"{baseline_config['task']} [baseline]"] = success
        except KeyboardInterrupt:
            print("\n\n⚠️  ユーザーによって中断されました")
            break
        
        # DR版
        try:
            dr_config = task_pair['dr']
            success = train_task(dr_config, 'DR')
            results[f"{dr_config['task']} [dr]"] = success
        except KeyboardInterrupt:
            print("\n\n⚠️  ユーザーによって中断されました")
            break
    
    # 結果サマリ
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("学習結果サマリ")
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
    
    print(f"\n開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"所要時間: {duration}")
    
    # 学習済みモデルの場所を表示
    if any(results.values()):
        print("\n" + "="*60)
        print("学習済みモデル")
        print("="*60)
        for task_pair in TASK_PAIRS:
            baseline_config = task_pair['baseline']
            dr_config = task_pair['dr']
            
            baseline_key = f"{baseline_config['task']} [baseline]"
            dr_key = f"{dr_config['task']} [dr]"
            
            if results.get(baseline_key, False) or results.get(dr_key, False):
                domain = baseline_config['task'].split('-')[0]
                print(f"\n  {domain.upper()}:")
                
                if results.get(baseline_key, False):
                    print(f"    Baseline: logs/{baseline_config['task']}/{SEED}/{baseline_config['exp_name']}/")
                
                if results.get(dr_key, False):
                    print(f"    DR:       logs/{dr_config['task']}/{SEED}/{dr_config['exp_name']}/")
    
    print("\n" + "="*60)
    
    # 全タスク成功なら0、失敗があれば1を返す
    return 0 if all(results.values()) else 1

if __name__ == '__main__':
    sys.exit(main())

