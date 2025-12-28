#!/usr/bin/env python3
"""Hopper Hop Backwards Zero-shot評価スクリプト

訓練済みモデルを5種類の固定大腿長で評価:
- 0.25m (0.76×)
- 0.33m (1.0×, baseline)
- 0.39m (1.18×)
- 0.43m (1.30×)
- 0.45m (1.36×)

各(model, seed, length)で30エピソード実行してIQMを計算。
"""

import subprocess
import sys
import os
import glob
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]

# 評価タスク設定
EVAL_TASKS = [
    ('hop_backwards_length_076x', 0.25, '0.76×'),
    ('hop_backwards_length_10x',  0.33, '1.0× (baseline)'),
    ('hop_backwards_length_12x',  0.39, '1.18×'),
    ('hop_backwards_length_13x',  0.43, '1.30×'),
    ('hop_backwards_length_136x', 0.45, '1.36×'),
]

# モデル設定
MODELS = [
    {
        'name': 'Baseline',
        'exp_name': 'hopper_baseline',
        'checkpoint_pattern': 'logs/hopper_baseline/*/checkpoint.pt',
    },
    {
        'name': 'DR',
        'exp_name': 'hopper_dr',
        'checkpoint_pattern': 'logs/hopper_dr/*/checkpoint.pt',
    },
    {
        'name': 'Model C',
        'exp_name': 'hopper_model_c',
        'checkpoint_pattern': 'logs/hopper_model_c/*/checkpoint.pt',
        'use_model_c': True,
        'c_phys_config': {
            'c_phys_dim': 1,
            'phys_param_type': 'length',
        }
    },
    {
        'name': 'Oracle',
        'exp_name': 'hopper_oracle',
        'checkpoint_pattern': 'logs/hopper_oracle/*/checkpoint.pt',
        'use_oracle': True,
        'c_phys_config': {
            'c_phys_dim': 1,
            'phys_param_type': 'length',
        }
    },
]

# 評価設定
EPISODES = 30
SEEDS = [0, 1, 2, 3, 4]


def find_checkpoint(model_config, seed):
    """seedに対応するチェックポイントを検索"""
    pattern = model_config['checkpoint_pattern']
    checkpoints = glob.glob(str(REPO_ROOT / pattern))
    
    # seedを含むパスを検索
    for ckpt in checkpoints:
        if f'seed{seed}' in ckpt or f'seed_{seed}' in ckpt or f'/{seed}/' in ckpt:
            return ckpt
    
    print(f"  Warning: Checkpoint not found for {model_config['name']} seed={seed}")
    return None


def evaluate_model(model_config, seed, task_name, length_value, length_label):
    """単一モデル・単一seed・単一タスクの評価"""
    model_name = model_config['name']
    
    # チェックポイント検索
    checkpoint = find_checkpoint(model_config, seed)
    if checkpoint is None:
        return False
    
    print(f"\n  Evaluating: {model_name} seed={seed} on {task_name}")
    print(f"    Checkpoint: {checkpoint}")
    print(f"    Thigh length: {length_value}m ({length_label})")
    
    # コマンド構築
    cmd = [
        sys.executable,
        'tdmpc2/evaluate.py',
        f'task=hopper-{task_name}',
        f'checkpoint={checkpoint}',
        f'episodes={EPISODES}',
        f'seed={seed}',
        'save_video=false',
    ]
    
    # Model C / Oracle の場合は物理パラメータ設定を追加
    if model_config.get('use_model_c'):
        cmd.append('use_model_c=true')
        c_phys_config = model_config['c_phys_config']
        cmd.extend([
            f"c_phys_dim={c_phys_config['c_phys_dim']}",
            f"phys_param_type={c_phys_config['phys_param_type']}",
        ])
    
    if model_config.get('use_oracle'):
        cmd.append('use_oracle=true')
        c_phys_config = model_config['c_phys_config']
        cmd.extend([
            f"c_phys_dim={c_phys_config['c_phys_dim']}",
            f"phys_param_type={c_phys_config['phys_param_type']}",
        ])
    
    # 実行
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=REPO_ROOT, 
            check=True, 
            env=env,
            capture_output=True,
            text=True
        )
        
        # 評価結果を抽出（標準出力から）
        output = result.stdout
        if 'episode_reward' in output or 'return' in output:
            print(f"    ✅ Completed")
            return True
        else:
            print(f"    ⚠️  Completed but no results found")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"    ❌ Failed")
        print(f"    Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n    ⚠️  Interrupted")
        raise


def main():
    """メイン処理"""
    start_time = datetime.now()
    
    print("="*80)
    print("Hopper Hop Backwards Zero-shot評価開始")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print(f"\nモデル数: {len(MODELS)}")
    print(f"Seeds: {SEEDS}")
    print(f"評価タスク数: {len(EVAL_TASKS)}")
    print(f"エピソード数/評価: {EPISODES}")
    
    total_evaluations = len(MODELS) * len(SEEDS) * len(EVAL_TASKS)
    print(f"合計評価数: {len(MODELS)} × {len(SEEDS)} × {len(EVAL_TASKS)} = {total_evaluations}")
    
    print("\n評価タスク:")
    for task_name, length, label in EVAL_TASKS:
        print(f"  - {task_name}: {length}m ({label})")
    
    print("\nモデル:")
    for model in MODELS:
        print(f"  - {model['name']}")
    print()
    
    # ベースディレクトリに移動
    os.chdir(REPO_ROOT)
    
    results = {}
    current_eval = 0
    
    # 各モデル × 各seed × 各タスク を評価
    for model in MODELS:
        model_name = model['name']
        results[model_name] = {}
        
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")
        
        for seed in SEEDS:
            results[model_name][seed] = {}
            
            print(f"\n--- Seed {seed} ---")
            
            for task_name, length, label in EVAL_TASKS:
                current_eval += 1
                print(f"\n[{current_eval}/{total_evaluations}]")
                
                try:
                    success = evaluate_model(
                        model, seed, task_name, length, label
                    )
                    results[model_name][seed][task_name] = success
                except KeyboardInterrupt:
                    print("\n\n⚠️  ユーザーによって中断されました")
                    break
    
    # 結果サマリ
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("評価結果サマリ")
    print("="*80)
    
    for model_name, seed_results in results.items():
        total = 0
        successes = 0
        
        for seed, task_results in seed_results.items():
            for task, success in task_results.items():
                total += 1
                if success:
                    successes += 1
        
        print(f"\n{model_name}: {successes}/{total} 成功")
        
        for seed in SEEDS:
            if seed in seed_results:
                seed_success = sum(1 for v in seed_results[seed].values() if v)
                seed_total = len(seed_results[seed])
                print(f"  seed={seed}: {seed_success}/{seed_total}")
    
    print(f"\n総実行時間: {duration}")
    print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\n次のステップ:")
    print("1. 結果CSVを結合: python merge_results.py")
    print("2. 統計解析: python analyze_results.py")
    print("3. プロット作成: python plot_learning_curves.py")


if __name__ == '__main__':
    main()

