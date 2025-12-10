#!/usr/bin/env python
"""
Baselineだけのパラメータ別性能グラフを作成

使用方法:
    python plot_baseline_only.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 既存のanalyze_results.pyの関数を流用
sys.path.insert(0, str(Path(__file__).parent / "evaluate"))
from analyze_results import compute_iqm, bootstrap_ci, load_and_preprocess, compute_iqm_per_combination

REPO_ROOT = Path(__file__).resolve().parent


def aggregate_baseline_only(df_iqm: pd.DataFrame) -> pd.DataFrame:
    """Baseline だけを集約"""
    df_baseline = df_iqm[df_iqm["model"] == "baseline"]
    
    aggregated = []
    for param, group in df_baseline.groupby("param"):
        iqm_values = group["iqm_return"].values
        
        mean_val = np.mean(iqm_values)
        ci_low, ci_high = bootstrap_ci(iqm_values, statistic_fn=np.mean)
        
        aggregated.append({
            "model": "baseline",
            "param": param,
            "mean": mean_val,
            "ci_low": ci_low,
            "ci_high": ci_high
        })
    
    return pd.DataFrame(aggregated)


def plot_baseline_only(df_agg: pd.DataFrame, output_path: Path):
    """Baseline だけをプロット"""
    print("="*70)
    print("Baseline のみ可視化")
    print("="*70)
    
    df_baseline = df_agg[df_agg["model"] == "baseline"].sort_values("param")
    
    params = df_baseline["param"].values
    means = df_baseline["mean"].values
    ci_lows = df_baseline["ci_low"].values
    ci_highs = df_baseline["ci_high"].values
    
    errors = [means - ci_lows, ci_highs - means]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(
        params, means, yerr=errors, 
        label="Baseline", marker='o', capsize=5, 
        linewidth=2, markersize=8, color='C0'
    )
    
    ax.set_xlabel("Mass Multiplier", fontsize=12)
    ax.set_ylabel("Mean IQM Return", fontsize=12)
    ax.set_title("Baseline Performance vs Mass (IQM with 95% CI)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # y軸の範囲を0から最大値+余白に設定
    ax.set_ylim(0, max(ci_highs) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"保存: {output_path}")
    print()


def main():
    csv_path = REPO_ROOT / "results.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} が見つかりません")
        print("まず evaluate/evaluate_all_models.py を実行してください")
        sys.exit(1)
    
    # データ読み込み
    df = load_and_preprocess(csv_path)
    
    # IQM計算
    df_iqm = compute_iqm_per_combination(df)
    
    # Baselineだけ集約
    df_agg = aggregate_baseline_only(df_iqm)
    
    print("Baseline性能:")
    print(df_agg.to_string(index=False))
    print()
    
    # プロット
    output_path = REPO_ROOT / "fig_baseline_only.png"
    plot_baseline_only(df_agg, output_path)
    
    print("="*70)
    print("完了")
    print("="*70)
    print(f"生成されたファイル: {output_path}")


if __name__ == "__main__":
    main()

