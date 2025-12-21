#!/usr/bin/env python3
"""Cheetah-Run 4モデル評価スクリプト

各モデルを異なる摩擦係数の環境で評価し、Zero-Shot性能を測定する。
"""

import subprocess
import sys
import os
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT_ENV = "TDMPC2_LOG_ROOT"
RUN_ROOT_ENV = "TDMPC2_RUN_ROOT"

# 評価する摩擦係数
FRICTION_VALUES = [0.2, 0.4, 0.6, 0.8]

# モデル設定（train_cheetah_all_models.pyと対応）
MODELS = [
    {
        'name': 'Baseline',
        'exp_name': 'cheetah_baseline',
        'train_task': 'cheetah-run_friction04',
    },
    {
        'name': 'DR',
        'exp_name': 'cheetah_dr',
        'train_task': 'cheetah-run_randomized',
    },
    {
        'name': 'Model C',
        'exp_name': 'cheetah_model_c',
        'exp_name_aliases': ['cheetah_modelc'],
        'train_task': 'cheetah-run_randomized',
    },
    {
        'name': 'Oracle',
        'exp_name': 'cheetah_oracle',
        'train_task': 'cheetah-run_randomized',
    },
]

# 評価パラメータ
SEEDS = [0, 1, 2, 3, 4]
NUM_EPISODES = 30  # 統計的有意性のため


def friction_to_task_name(friction):
    """摩擦係数から評価タスク名を生成
    
    Examples:
        0.2 -> cheetah-run_friction02
        0.4 -> cheetah-run_friction04
        0.6 -> cheetah-run_friction06
        0.8 -> cheetah-run_friction08
    """
    # 小数点以下1桁を2桁の整数に変換（0.2 -> 02, 0.4 -> 04）
    friction_int = int(friction * 10)
    return f"cheetah-run_friction{friction_int:02d}"


def get_log_roots():
    roots = []
    log_root_env = os.environ.get(LOG_ROOT_ENV)
    if log_root_env:
        roots.append(Path(log_root_env))
    run_root_env = os.environ.get(RUN_ROOT_ENV)
    if run_root_env:
        roots.append(Path(run_root_env) / "logs")
    roots.extend([
        REPO_ROOT / "logs",
        Path.cwd() / "logs",
        REPO_ROOT.parent / "logs",
        Path.home() / "tdmpc3" / "tdmpc3" / "logs",
        REPO_ROOT / "logs_remote",
    ])
    seen = set()
    unique = []
    for root in roots:
        root = Path(root)
        if root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def find_checkpoint(task, exp_name, seed, exp_name_aliases=None):
    """チェックポイントを探す
    
    リモートサーバーのディレクトリ構造:
    ~/tdmpc3/tdmpc3/logs/
    ├── cheetah-run_friction04/
    │   └── 0/cheetah_baseline/models/final.pt
    └── cheetah-run_randomized/
        └── 0/
            ├── cheetah_dr/models/final.pt
            ├── cheetah_modelc/models/final.pt
            └── cheetah_oracle/models/final.pt
    """
    # 優先順位の高い順に探索
    exp_names = [exp_name]
    if exp_name_aliases:
        exp_names.extend([name for name in exp_name_aliases if name not in exp_names])

    search_paths = []
    for log_root in get_log_roots():
        for name in exp_names:
            search_paths.append(log_root / task / str(seed) / name / "models" / "final.pt")
    for name in exp_names:
        search_paths.append(REPO_ROOT / "checkpoints" / f"{name}_seed{seed}.pt")
    
    for path in search_paths:
        if path.exists():
            print(f"  Found checkpoint: {path}")
            return path
    
    print(f"  ❌ Checkpoint not found for {exp_name} seed={seed}")
    print(f"  Searched paths:")
    for path in search_paths:
        print(f"    - {path}")
    
    return None


def evaluate_model(model_config, seed, friction):
    """単一モデル・単一seed・単一摩擦係数で評価"""
    name = model_config['name']
    exp_name = model_config['exp_name']
    train_task = model_config['train_task']
    eval_task = friction_to_task_name(friction)
    
    # チェックポイントを探す
    checkpoint = find_checkpoint(
        train_task,
        exp_name,
        seed,
        exp_name_aliases=model_config.get('exp_name_aliases'),
    )
    if checkpoint is None:
        print(f"❌ Checkpoint not found: {name} seed={seed}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {name} (seed={seed}, friction={friction})")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Eval task: {eval_task}")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"{'='*70}")
    
    # 評価コマンド
    cmd = [
        sys.executable,
        'tdmpc2/evaluate.py',
        f'task={eval_task}',
        f'checkpoint={checkpoint}',
        f'episodes={NUM_EPISODES}',
        f'seed={seed}',
        'save_video=false',
    ]
    
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
        
        # 標準出力から平均リターンを抽出
        output = result.stdout
        for line in output.split('\n'):
            if 'Average return' in line or 'Mean return' in line:
                # 例: "Average return: 850.5"
                try:
                    avg_return = float(line.split(':')[-1].strip())
                    print(f"✅ Avg return: {avg_return:.2f}")
                    return avg_return
                except:
                    pass
        
        print(f"⚠️  Could not parse return from output")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {name} seed={seed} friction={friction}")
        print(f"Error: {e}")
        return None
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted: {name} seed={seed} friction={friction}")
        raise


def main():
    """メイン処理"""
    start_time = datetime.now()
    
    print("="*70)
    print("Cheetah-Run 4モデル評価開始")
    print(f"開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print(f"\nモデル数: {len(MODELS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Friction values: {FRICTION_VALUES}")
    print(f"Episodes per eval: {NUM_EPISODES}")
    print(f"合計評価数: {len(MODELS)} × {len(SEEDS)} × {len(FRICTION_VALUES)} = {len(MODELS) * len(SEEDS) * len(FRICTION_VALUES)}")
    print()
    
    os.chdir(REPO_ROOT)
    
    # 結果を保存するリスト
    results = []
    
    total_evals = len(MODELS) * len(SEEDS) * len(FRICTION_VALUES)
    current_eval = 0
    
    # 各モデル × 各seed × 各摩擦係数を評価
    for model in MODELS:
        model_name = model['name']
        
        for seed in SEEDS:
            for friction in FRICTION_VALUES:
                current_eval += 1
                
                print(f"\n{'#'*70}")
                print(f"進捗: {current_eval}/{total_evals}")
                print(f"モデル: {model_name}, seed={seed}, friction={friction}")
                print(f"{'#'*70}")
                
                try:
                    avg_return = evaluate_model(model, seed, friction)
                    
                    results.append({
                        'model': model_name,
                        'seed': seed,
                        'friction': friction,
                        'avg_return': avg_return,
                    })
                    
                except KeyboardInterrupt:
                    print("\n\n⚠️  ユーザーによって中断されました")
                    break
    
    # 結果をCSVに保存
    results_df = pd.DataFrame(results)
    output_csv = REPO_ROOT / "cheetah_evaluation_results.csv"
    results_df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*70}")
    print("評価完了")
    print(f"{'='*70}")
    print(f"結果を保存: {output_csv}")
    
    # サマリ表示
    if len(results) > 0:
        print("\n結果サマリ（平均リターン）:")
        print("="*70)
        
        summary = results_df.pivot_table(
            index='model',
            columns='friction',
            values='avg_return',
            aggfunc='mean'
        )
        print(summary.to_string())
        
        print(f"\n総実行時間: {datetime.now() - start_time}")
        print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("評価結果がありません")
    
    print("="*70)


if __name__ == '__main__':
    main()
