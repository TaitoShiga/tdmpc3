"""
学習曲線（サンプル効率）を可視化するスクリプト

logs_remote/から各モデルのeval.csvを読み込み、
学習曲線をプロットして eval_curves.png に保存する。
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import seaborn as sns

# 見た目の設定
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

# モデル名とラベルのマッピング
LABELS = {
    "baseline": "Baseline",
    "dr": "DR",
    "c": "Model C",
    "o": "Oracle"
}

COLORS = {
    "baseline": "#1f77b4",
    "dr": "#ff7f0e",
    "c": "#2ca02c",
    "o": "#d62728"
}


def load_eval_csv(model: str, seed: int, logs_dir: Path) -> pd.DataFrame:
    """指定されたモデルとseedのeval.csvを読み込む"""
    if model == "baseline":
        csv_path = logs_dir / "pendulum-swingup" / str(seed) / "baseline" / "eval.csv"
    elif model == "dr":
        csv_path = logs_dir / "pendulum-swingup-randomized" / str(seed) / "dr" / "eval.csv"
    elif model == "c":
        csv_path = logs_dir / "pendulum-swingup-randomized" / str(seed) / "modelc" / "eval.csv"
    elif model == "o":
        csv_path = logs_dir / "pendulum-swingup-randomized" / str(seed) / "oracle" / "eval.csv"
    else:
        raise ValueError(f"Unknown model: {model}")
    
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found.")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path.name} with {len(df)} rows (seed={seed})")
    return df


def compute_statistics(dfs: List[pd.DataFrame]) -> tuple:
    """複数seedのデータから平均と95%信頼区間を計算"""
    # 全seedのステップを統一
    min_length = min(len(df) for df in dfs)
    aligned_dfs = [df.iloc[:min_length].reset_index(drop=True) for df in dfs]
    
    # 報酬を抽出
    rewards = np.array([df['episode_reward'].values for df in aligned_dfs])
    steps = aligned_dfs[0]['step'].values
    
    # 統計量を計算
    means = np.mean(rewards, axis=0)
    stds = np.std(rewards, axis=0)
    ci_lows = np.percentile(rewards, 2.5, axis=0)
    ci_highs = np.percentile(rewards, 97.5, axis=0)
    
    return steps, means, stds, ci_lows, ci_highs


def plot_learning_curves(data: Dict[str, List[pd.DataFrame]], output_path: Path):
    """学習曲線をプロット"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model in ["baseline", "dr", "c", "o"]:
        if model not in data or len(data[model]) == 0:
            print(f"Skipping {model}: no data")
            continue
        
        steps, means, stds, ci_lows, ci_highs = compute_statistics(data[model])
        
        # メインライン
        ax.plot(steps, means, label=LABELS[model], color=COLORS[model], 
                linewidth=2.5, alpha=0.9)
        
        # 信頼区間
        ax.fill_between(steps, ci_lows, ci_highs, color=COLORS[model], alpha=0.2)
    
    ax.set_xlabel('Environment Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Episode Return', fontsize=13, fontweight='bold')
    ax.set_title('Learning Curves: Sample Efficiency Comparison', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved learning curves to {output_path}")
    plt.show()


def print_summary(data: Dict[str, List[pd.DataFrame]]):
    """最終性能と収束ステップのサマリーを出力"""
    print("\n" + "="*80)
    print("LEARNING EFFICIENCY SUMMARY")
    print("="*80)
    
    for model in ["baseline", "dr", "c", "o"]:
        if model not in data or len(data[model]) == 0:
            continue
        
        print(f"\n{LABELS[model]}:")
        print("-" * 40)
        
        # 最終性能（最後の1000ステップの平均）
        final_rewards = []
        for df in data[model]:
            window = 1000  # 平滑化ウィンドウ
            if len(df) >= window:
                final_rewards.append(df['episode_reward'].iloc[-window:].mean())
            else:
                final_rewards.append(df['episode_reward'].mean())
        
        mean_final = np.mean(final_rewards)
        std_final = np.std(final_rewards)
        print(f"  Final Performance: {mean_final:.1f} ± {std_final:.1f}")
        
        # 収束ステップ（平均報酬が最終性能の95%に達したステップ）
        convergence_steps = []
        for df in data[model]:
            target = df['episode_reward'].iloc[-window:].mean() * 0.95 if len(df) >= window else df['episode_reward'].mean() * 0.95
            converged = df[df['episode_reward'] >= target]
            if len(converged) > 0:
                convergence_steps.append(converged.iloc[0]['step'])
        
        if convergence_steps:
            mean_conv = np.mean(convergence_steps)
            std_conv = np.std(convergence_steps)
            print(f"  Convergence Step (95%): {mean_conv:.0f} ± {std_conv:.0f}")
        else:
            print(f"  Convergence Step (95%): N/A")


def main():
    logs_dir = Path("logs_remote")
    output_path = Path("eval_curves.png")
    
    if not logs_dir.exists():
        print(f"Error: {logs_dir} not found!")
        return
    
    # データを読み込む
    data = {
        "baseline": [],
        "dr": [],
        "c": [],
        "o": []
    }
    
    num_seeds = 5
    for seed in range(num_seeds):
        for model in ["baseline", "dr", "c", "o"]:
            df = load_eval_csv(model, seed, logs_dir)
            if df is not None:
                data[model].append(df)
    
    # データが存在するか確認
    if all(len(v) == 0 for v in data.values()):
        print("Error: No data loaded!")
        return
    
    # プロット
    plot_learning_curves(data, output_path)
    
    # サマリー出力
    print_summary(data)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()

