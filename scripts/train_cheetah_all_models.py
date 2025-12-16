#!/usr/bin/env python3
"""Cheetah-Run 4モデル訓練スクリプト

Baseline, DR, Model C, Oracleの4モデルを学習する。
各モデルを5 seeds (0-4) で訓練。
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# モデル設定
MODELS = [
    {
        'name': 'Baseline',
        'task': 'cheetah-run_friction04',  # 固定摩擦 (0.4)
        'exp_name': 'cheetah_baseline',
        'use_oracle': False,
        'use_model_c': False,
        'steps': 1000000,
    },
    {
        'name': 'DR',
        'task': 'cheetah-run_randomized',  # ランダム摩擦 (0.2-0.8)
        'exp_name': 'cheetah_dr',
        'use_oracle': False,
        'use_model_c': False,
        'steps': 1000000,
    },
    {
        'name': 'Model C',
        'task': 'cheetah-run_randomized',
        'exp_name': 'cheetah_model_c',
        'use_oracle': False,
        'use_model_c': True,
        'steps': 1000000,
        'gru_pretrained': None,  # GRUは学習中にゼロから訓練
    },
    {
        'name': 'Oracle',
        'task': 'cheetah-run_randomized',
        'exp_name': 'cheetah_oracle',
        'use_oracle': True,
        'use_model_c': False,
        'steps': 1000000,
    },
]

# 共通パラメータ
SEEDS = [0, 1, 2, 3, 4]
SAVE_VIDEO = False

# Model C / Oracle 用の物理パラメータ設定
C_PHYS_CONFIG = {
    'c_phys_dim': 1,  # 摩擦係数1次元
    'phys_param_type': 'friction',
    'phys_param_normalization': 'standard',
    'context_length': 50,
    'gru_hidden_dim': 256,
}


def train_model(model_config, seed):
    """単一モデル・単一seedの学習"""
    name = model_config['name']
    task = model_config['task']
    exp_name = model_config['exp_name']
    steps = model_config['steps']
    
    print("\n" + "="*70)
    print(f"Training: {name} (seed={seed})")
    print(f"  Task: {task}")
    print(f"  Exp name: {exp_name}")
    print(f"  Steps: {steps}")
    print("="*70)
    
    # コマンド構築
    cmd = [
        sys.executable,
        'tdmpc2/train.py',
        f'task={task}',
        f'exp_name={exp_name}',
        f'steps={steps}',
        f'seed={seed}',
        f'save_video={str(SAVE_VIDEO).lower()}',
        'enable_wandb=false',
        'compile=false',
    ]
    
    # Oracle/Model C 用の設定を追加
    if model_config['use_oracle']:
        cmd.append('use_oracle=true')
        cmd.extend([
            f"c_phys_dim={C_PHYS_CONFIG['c_phys_dim']}",
            f"phys_param_type={C_PHYS_CONFIG['phys_param_type']}",
            f"phys_param_normalization={C_PHYS_CONFIG['phys_param_normalization']}",
        ])
    
    if model_config['use_model_c']:
        cmd.append('use_model_c=true')
        cmd.extend([
            f"c_phys_dim={C_PHYS_CONFIG['c_phys_dim']}",
            f"phys_param_type={C_PHYS_CONFIG['phys_param_type']}",
            f"phys_param_normalization={C_PHYS_CONFIG['phys_param_normalization']}",
            f"context_length={C_PHYS_CONFIG['context_length']}",
            f"gru_hidden_dim={C_PHYS_CONFIG['gru_hidden_dim']}",
        ])
        
        # 事前学習GRUがあれば追加
        if model_config.get('gru_pretrained'):
            cmd.append(f"gru_pretrained={model_config['gru_pretrained']}")
    
    # 実行
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    
    try:
        start = datetime.now()
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
        duration = datetime.now() - start
        
        print(f"✅ Completed: {name} (seed={seed}) in {duration}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {name} (seed={seed})")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted: {name} (seed={seed})")
        raise


def main():
    """メイン処理"""
    start_time = datetime.now()
    
    print("="*70)
    print("Cheetah-Run 4モデル連続学習開始")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"\nモデル数: {len(MODELS)}")
    print(f"Seeds: {SEEDS}")
    print(f"合計実行数: {len(MODELS)} × {len(SEEDS)} = {len(MODELS) * len(SEEDS)}")
    print("\nモデル:")
    for model in MODELS:
        print(f"  - {model['name']}: {model['task']}")
    print()
    
    # ベースディレクトリに移動
    os.chdir(REPO_ROOT)
    
    results = {}
    total_runs = len(MODELS) * len(SEEDS)
    current_run = 0
    
    # 各モデル × 各seed を学習
    for model in MODELS:
        model_name = model['name']
        results[model_name] = {}
        
        for seed in SEEDS:
            current_run += 1
            print(f"\n{'#'*70}")
            print(f"進捗: {current_run}/{total_runs} - {model_name} seed={seed}")
            print(f"{'#'*70}")
            
            try:
                success = train_model(model, seed)
                results[model_name][seed] = success
            except KeyboardInterrupt:
                print("\n\n⚠️  ユーザーによって中断されました")
                break
    
    # 結果サマリ
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("学習結果サマリ")
    print("="*70)
    
    for model_name, seed_results in results.items():
        successes = sum(1 for v in seed_results.values() if v)
        total = len(seed_results)
        print(f"{model_name}: {successes}/{total} 成功")
        
        for seed, success in seed_results.items():
            status = "✅" if success else "❌"
            print(f"  seed={seed}: {status}")
    
    print(f"\n総実行時間: {duration}")
    print(f"終了時刻: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()

