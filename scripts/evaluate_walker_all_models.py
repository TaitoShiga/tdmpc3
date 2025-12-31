#!/usr/bin/env python3
"""Walker Walk Zero-shot評価スクリプト

訓練済みモデル（seed0のみ）を8種類の固定胴体質量で評価:

In-Distribution (訓練範囲内: 0.5x-2.5x):
- 0.5x = 1.67 kg
- 1.0x = 3.34 kg (baseline)
- 1.5x = 5.01 kg
- 2.0x = 6.68 kg
- 2.5x = 8.35 kg

Out-of-Distribution (訓練範囲外):
- 0.3x = 1.00 kg (軽すぎ)
- 3.0x = 10.02 kg (重すぎ)
- 3.5x = 11.69 kg (極端)

各(model, mass)で30エピソード実行してIQMを計算。
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]

# 評価タスク設定
EVAL_TASKS = [
    # In-Distribution
    ('walk_torso_mass_05x', 0.5, 1.67, 'In-Dist'),
    ('walk_torso_mass_10x', 1.0, 3.34, 'In-Dist (baseline)'),
    ('walk_torso_mass_15x', 1.5, 5.01, 'In-Dist'),
    ('walk_torso_mass_20x', 2.0, 6.68, 'In-Dist'),
    ('walk_torso_mass_25x', 2.5, 8.35, 'In-Dist'),
    # Out-of-Distribution
    ('walk_torso_mass_03x', 0.3, 1.00, 'OOD (light)'),
    ('walk_torso_mass_30x', 3.0, 10.02, 'OOD (heavy)'),
    ('walk_torso_mass_35x', 3.5, 11.69, 'OOD (extreme)'),
]

# モデル設定（seed0のみ）
MODELS = [
    {
        'name': 'Baseline',
        'exp_name': 'walker_baseline',
        'checkpoint_path': 'logs/walker-walk/0/walker_baseline/models/final.pt',
    },
    {
        'name': 'DR',
        'exp_name': 'walker_dr',
        'checkpoint_path': 'logs/walker-walk_randomized/0/walker_dr/models/final.pt',
    },
    {
        'name': 'Model C',
        'exp_name': 'walker_model_c',
        'checkpoint_path': 'logs/walker-walk_randomized/0/walker_model_c/models/final.pt',
        'use_model_c': True,
        'c_phys_config': {
            'c_phys_dim': 1,
            'phys_param_type': 'mass',
            'phys_param_normalization': 'standard',
            'context_length': 50,
            'gru_hidden_dim': 256,
        }
    },
    {
        'name': 'Oracle',
        'exp_name': 'walker_oracle',
        'checkpoint_path': 'logs/walker-walk_randomized/0/walker_oracle/models/final.pt',
        'use_oracle': True,
        'c_phys_config': {
            'c_phys_dim': 1,
            'phys_param_type': 'mass',
            'phys_param_normalization': 'standard',
        }
    },
]

# 評価設定
EPISODES = 30
SEED = 0


def find_checkpoint(model_config):
    """seed0のチェックポイントを取得"""
    checkpoint_path = REPO_ROOT / model_config['checkpoint_path']
    
    if not checkpoint_path.exists():
        print(f"  Warning: Checkpoint not found: {checkpoint_path}")
        return None
    
    return str(checkpoint_path)


def evaluate_model(model_config, task_name, multiplier, mass_kg, dist_label):
    """単一モデル・単一タスクの評価"""
    model_name = model_config['name']
    
    # チェックポイント検索
    checkpoint = find_checkpoint(model_config)
    if checkpoint is None:
        return False
    
    print(f"\n  Evaluating: {model_name} on {task_name}")
    print(f"    Checkpoint: {checkpoint}")
    print(f"    Torso mass: {multiplier:.1f}x = {mass_kg:.2f} kg ({dist_label})")
    
    # コマンド構築
    cmd = [
        sys.executable,
        'tdmpc2/evaluate.py',
        f'task=walker-{task_name}',
        f'checkpoint={checkpoint}',
        f'episodes={EPISODES}',
        f'seed={SEED}',
        'save_video=false',
    ]
    
    # Model C / Oracle の場合は物理パラメータ設定を追加
    if model_config.get('use_model_c'):
        cmd.append('use_model_c=true')
        c_phys_config = model_config['c_phys_config']
        cmd.extend([
            f"c_phys_dim={c_phys_config['c_phys_dim']}",
            f"phys_param_type={c_phys_config['phys_param_type']}",
            f"phys_param_normalization={c_phys_config['phys_param_normalization']}",
            f"context_length={c_phys_config['context_length']}",
            f"gru_hidden_dim={c_phys_config['gru_hidden_dim']}",
        ])
    
    if model_config.get('use_oracle'):
        cmd.append('use_oracle=true')
        c_phys_config = model_config['c_phys_config']
        cmd.extend([
            f"c_phys_dim={c_phys_config['c_phys_dim']}",
            f"phys_param_type={c_phys_config['phys_param_type']}",
            f"phys_param_normalization={c_phys_config['phys_param_normalization']}",
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
        print(f"    Stdout: {e.stdout}")
        print(f"    Stderr: {e.stderr}")
        return False
    except KeyboardInterrupt:
        print(f"\n    ⚠️  Interrupted")
        raise


def main():
    """メイン処理"""
    start_time = datetime.now()
    
    print("="*80)
    print("Walker Walk Zero-shot評価開始 (seed0のみ)")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print(f"\nモデル数: {len(MODELS)}")
    print(f"Seed: {SEED}")
    print(f"評価タスク数: {len(EVAL_TASKS)}")
    print(f"エピソード数/評価: {EPISODES}")
    
    total_evaluations = len(MODELS) * len(EVAL_TASKS)
    print(f"合計評価数: {len(MODELS)} × {len(EVAL_TASKS)} = {total_evaluations}")
    
    print("\n評価タスク:")
    print("  In-Distribution (訓練範囲: 0.5x-2.5x):")
    for task_name, mult, mass, label in EVAL_TASKS:
        if 'In-Dist' in label:
            print(f"    - {mult:.1f}x = {mass:.2f} kg ({label})")
    print("  Out-of-Distribution (訓練範囲外):")
    for task_name, mult, mass, label in EVAL_TASKS:
        if 'OOD' in label:
            print(f"    - {mult:.1f}x = {mass:.2f} kg ({label})")
    
    print("\nモデル:")
    for model in MODELS:
        print(f"  - {model['name']}")
    print()
    
    # ベースディレクトリに移動
    os.chdir(REPO_ROOT)
    
    results = {}
    current_eval = 0
    
    # 各モデル × 各タスク を評価
    for model in MODELS:
        model_name = model['name']
        results[model_name] = {}
        
        print(f"\n{'='*80}")
        print(f"Model: {model_name}")
        print(f"{'='*80}")
        
        for task_name, mult, mass, label in EVAL_TASKS:
            current_eval += 1
            print(f"\n[{current_eval}/{total_evaluations}]")
            
            try:
                success = evaluate_model(
                    model, task_name, mult, mass, label
                )
                results[model_name][task_name] = success
            except KeyboardInterrupt:
                print("\n\n⚠️  ユーザーによって中断されました")
                break
        
        # モデル間で中断された場合は全体を終了
        if current_eval < total_evaluations and task_name != EVAL_TASKS[-1][0]:
            break
    
    # 結果サマリー
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("評価完了")
    print("="*80)
    print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"所要時間: {duration}")
    
    print("\n結果サマリー:")
    for model_name, tasks in results.items():
        success_count = sum(1 for v in tasks.values() if v)
        total_count = len(tasks)
        print(f"  {model_name}: {success_count}/{total_count} 成功")
    
    print("\n✅ 全評価完了")
    print(f"\n結果は各モデルのログディレクトリに保存されています:")
    print(f"  logs/<exp_name>/seed0*/eval_*.csv")


if __name__ == '__main__':
    main()

