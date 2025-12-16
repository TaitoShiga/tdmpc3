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
    """摩擦係数から評価タスク名を生成"""
    friction_str = f"{int(friction * 10):02d}"
    return f"cheetah-run_friction{friction_str}"


def find_checkpoint(task, exp_name, seed):
    """チェックポイントを探す"""
    # 優先順位の高い順に探索
    search_paths = [
        REPO_ROOT / "logs" / task / str(seed) / exp_name / "models" / "final.pt",
        REPO_ROOT / "checkpoints" / f"{exp_name}_seed{seed}.pt",
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def evaluate_model(model_config, seed, friction):
    """単一モデル・単一seed・単一摩擦係数で評価"""
    name = model_config['name']
    exp_name = model_config['exp_name']
    train_task = model_config['train_task']
    eval_task = friction_to_task_name(friction)
    
    # チェックポイントを探す
    checkpoint = find_checkpoint(train_task, exp_name, seed)
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
        f'model={checkpoint}',
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

