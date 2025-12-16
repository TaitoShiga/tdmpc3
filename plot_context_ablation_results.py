#!/usr/bin/env python
"""
Context Length Ablationの結果を可視化

使用方法:
    python plot_context_ablation_results.py --seed 0
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(seed, context_lengths=[10, 25, 50, 100, 200]):
    """学習曲線をプロット"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(context_lengths)))
    
    for ctx_len, color in zip(context_lengths, colors):
        path = f"logs/pendulum-swingup-randomized/{seed}/modelc_ctx{ctx_len}/eval.csv"
        
        try:
            df = pd.read_csv(path)
            ax.plot(df["step"], df["episode_reward"], 
                   label=f"ctx={ctx_len}", linewidth=2, color=color)
            print(f"✓ Loaded: ctx={ctx_len}, {len(df)} points")
        except FileNotFoundError:
            print(f"✗ Not found: {path}")
    
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title(f"Learning Curves: Context Length Ablation (seed={seed})", 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = f"context_ablation_curves_seed{seed}.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n保存: {output_path}")
    plt.show()


def compare_final_performance(seeds=[0, 1, 2], context_lengths=[10, 25, 50, 100, 200]):
    """最終性能を比較"""
    results = {}
    
    for ctx_len in context_lengths:
        returns = []
        for seed in seeds:
            path = f"logs/pendulum-swingup-randomized/{seed}/modelc_ctx{ctx_len}/eval.csv"
            try:
                df = pd.read_csv(path)
                # 最後10%の平均
                final_return = df["episode_reward"].tail(int(len(df) * 0.1)).mean()
                returns.append(final_return)
            except FileNotFoundError:
                print(f"✗ Not found: ctx={ctx_len}, seed={seed}")
        
        if returns:
            results[ctx_len] = {
                "mean": np.mean(returns),
                "std": np.std(returns),
                "n_seeds": len(returns)
            }
    
    # 表示
    print("\n" + "="*60)
    print("最終性能比較（最後10%の平均）")
    print("="*60)
    print(f"{'Context':>8} | {'Mean':>8} | {'Std':>8} | {'Seeds':>6}")
    print("-"*60)
    
    for ctx_len in context_lengths:
        if ctx_len in results:
            r = results[ctx_len]
            print(f"{ctx_len:>8} | {r['mean']:>8.1f} | {r['std']:>8.1f} | {r['n_seeds']:>6}")
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ctx_lens = list(results.keys())
    means = [results[c]["mean"] for c in ctx_lens]
    stds = [results[c]["std"] for c in ctx_lens]
    
    ax.errorbar(ctx_lens, means, yerr=stds, 
               marker='o', capsize=5, linewidth=2, markersize=8)
    
    ax.set_xlabel("Context Length", fontsize=12)
    ax.set_ylabel("Final Performance (Episode Return)", fontsize=12)
    ax.set_title("Context Length Ablation: Final Performance", 
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # 最適なcontext lengthをハイライト
    best_ctx = max(results.keys(), key=lambda k: results[k]["mean"])
    best_mean = results[best_ctx]["mean"]
    ax.axhline(best_mean, color='red', linestyle='--', alpha=0.5, 
              label=f'Best: ctx={best_ctx}')
    ax.legend()
    
    plt.tight_layout()
    output_path = "context_ablation_final_performance.png"
    plt.savefig(output_path, dpi=150)
    print(f"\n保存: {output_path}")
    plt.show()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                       help="学習曲線を表示するseed（デフォルト: 0）")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                       help="最終性能比較に使用するseeds（デフォルト: 0 1 2）")
    parser.add_argument("--context-lengths", type=int, nargs="+",
                       default=[10, 25, 50, 100, 200],
                       help="比較するcontext lengths")
    args = parser.parse_args()
    
    print("="*60)
    print("Context Length Ablation - 結果分析")
    print("="*60)
    
    # 1. 学習曲線
    print(f"\n1. 学習曲線（seed={args.seed}）")
    print("-"*60)
    plot_learning_curves(args.seed, args.context_lengths)
    
    # 2. 最終性能比較
    print(f"\n2. 最終性能比較（seeds={args.seeds}）")
    print("-"*60)
    results = compare_final_performance(args.seeds, args.context_lengths)
    
    print("\n" + "="*60)
    print("分析完了！")
    print("="*60)


if __name__ == "__main__":
    main()

