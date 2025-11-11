#!/usr/bin/env python3
"""Baseline vs DR 学習曲線比較スクリプト"""

import pandas as pd
import os
from pathlib import Path

# タスク設定
TASK_PAIRS = [
    ('pendulum-swingup', 'pendulum-swingup-randomized', 'pendulum'),
    ('cup-catch', 'cup-catch-randomized', 'ball_in_cup'),
    ('reacher-three-easy', 'reacher-three-easy-randomized', 'reacher'),
    ('hopper-stand', 'hopper-stand-randomized', 'hopper'),
]

SEED = 0

def load_eval_csv(task, exp_name):
    """eval.csvを読み込む"""
    # /tdmpc2/tdmpc2/logs/{task}/{seed}/{exp_name}/eval.csv
    csv_path = f"/tdmpc2/tdmpc2/logs/{task}/{SEED}/{exp_name}/eval.csv"
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"  Error loading {csv_path}: {e}")
        return None

def compare_tasks():
    """全タスクのBaseline vs DRを比較"""
    print("="*60)
    print("Baseline vs DR 学習曲線比較")
    print("="*60)
    
    results = []
    
    for baseline_task, dr_task, domain in TASK_PAIRS:
        print(f"\n{'='*60}")
        print(f"{domain.upper()}")
        print(f"{'='*60}")
        
        # Baseline
        baseline_df = load_eval_csv(baseline_task, f'{domain}_baseline')
        # DR
        dr_df = load_eval_csv(dr_task, f'{domain}_dr')
        
        if baseline_df is None and dr_df is None:
            print("  ⚠️  データなし（両方）")
            continue
        elif baseline_df is None:
            print("  ⚠️  Baselineデータなし")
            print(f"  DR: {len(dr_df)} episodes")
            continue
        elif dr_df is None:
            print("  ⚠️  DRデータなし")
            print(f"  Baseline: {len(baseline_df)} episodes")
            continue
        
        # 統計計算
        baseline_final = baseline_df['episode_reward'].iloc[-5:].mean() if len(baseline_df) >= 5 else baseline_df['episode_reward'].iloc[-1]
        dr_final = dr_df['episode_reward'].iloc[-5:].mean() if len(dr_df) >= 5 else dr_df['episode_reward'].iloc[-1]
        
        baseline_max = baseline_df['episode_reward'].max()
        dr_max = dr_df['episode_reward'].max()
        
        print(f"\n  Baseline:")
        print(f"    Episodes: {len(baseline_df)}")
        print(f"    Final reward (avg last 5): {baseline_final:.2f}")
        print(f"    Max reward: {baseline_max:.2f}")
        
        print(f"\n  DR:")
        print(f"    Episodes: {len(dr_df)}")
        print(f"    Final reward (avg last 5): {dr_final:.2f}")
        print(f"    Max reward: {dr_max:.2f}")
        
        print(f"\n  Comparison:")
        diff = dr_final - baseline_final
        pct_change = (diff / baseline_final * 100) if baseline_final != 0 else 0
        
        if abs(diff) < 1:
            status = "≈ 同等"
        elif diff > 0:
            status = f"↑ DR が優位 (+{diff:.2f}, +{pct_change:.1f}%)"
        else:
            status = f"↓ Baseline が優位 ({diff:.2f}, {pct_change:.1f}%)"
        
        print(f"    {status}")
        
        results.append({
            'domain': domain,
            'baseline_final': baseline_final,
            'dr_final': dr_final,
            'diff': diff,
            'pct_change': pct_change,
        })
    
    # サマリテーブル
    if results:
        print(f"\n{'='*60}")
        print("サマリ")
        print(f"{'='*60}\n")
        
        print(f"{'Domain':<15} {'Baseline':<12} {'DR':<12} {'Diff':<10} {'Change':<10}")
        print("-" * 60)
        
        for r in results:
            print(f"{r['domain']:<15} {r['baseline_final']:>10.2f}  {r['dr_final']:>10.2f}  "
                  f"{r['diff']:>8.2f}  {r['pct_change']:>7.1f}%")
    
    print(f"\n{'='*60}")

def main():
    # ベースディレクトリに移動
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, '..'))
    
    try:
        compare_tasks()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())

