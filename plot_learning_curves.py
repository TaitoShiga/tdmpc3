"""
学習曲線（サンプル効率）を可視化するスクリプト

logs_remote/から各モデルのeval.csvを読み込み（pendulum）、
または artifacts/から読み込み（cheetah）、
学習曲線をプロットして eval_curves.png に保存する。

Usage:
    # Pendulum（デフォルト）
    python plot_learning_curves.py
    
    # Cheetah
    python plot_learning_curves.py --task cheetah
    
    # 移動平均でスムージング
    python plot_learning_curves.py --task cheetah --smooth moving_average --window 10
    
    # ガウシアンフィルタ
    python plot_learning_curves.py --task pendulum --smooth gaussian --window 5
    
    # 指数移動平均
    python plot_learning_curves.py --task cheetah --smooth ema --alpha 0.1
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import seaborn as sns
import argparse
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

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

TASK_TITLES = {
    "pendulum": "Pendulum",
    "cheetah": "Cheetah",
    "walker_actuator": "Walker Actuator",
}


def load_eval_csv(model: str, seed: int, logs_dir: Path, task: str = "pendulum") -> pd.DataFrame:
    """指定されたモデルとseedのeval.csvを読み込む
    
    Args:
        model: "baseline", "dr", "c", or "o"
        seed: seed番号
        logs_dir: ログディレクトリ（pendulum用）
        task: "pendulum", "cheetah", or "walker_actuator"
    """
    if task == "cheetah":
        # artifactsディレクトリから読み込む
        artifacts_dir = Path("artifacts")
        if model == "baseline":
            csv_path = artifacts_dir / "cheetah_baseline" / f"eval_seed{seed}.csv"
        elif model == "dr":
            csv_path = artifacts_dir / "cheetah_dr" / f"eval_seed{seed}.csv"
        elif model == "c":
            csv_path = artifacts_dir / "cheetah_c" / f"eval_seed{seed}.csv"
        elif model == "o":
            csv_path = artifacts_dir / "cheetah_oracle" / f"eval_seed{seed}.csv"
        else:
            raise ValueError(f"Unknown model: {model}")
    elif task == "walker_actuator":
        if model == "baseline":
            csv_path = logs_dir / "walker-walk" / str(seed) / "walker_baseline" / "eval.csv"
        elif model == "dr":
            csv_path = logs_dir / "walker-walk_actuator_randomized" / str(seed) / "walker_actuator_dr" / "eval.csv"
        elif model == "c":
            csv_path = logs_dir / "walker-walk_actuator_randomized" / str(seed) / "walker_actuator_model_c" / "eval.csv"
        elif model == "o":
            csv_path = logs_dir / "walker-walk_actuator_randomized" / str(seed) / "walker_actuator_oracle" / "eval.csv"
        else:
            raise ValueError(f"Unknown model: {model}")
    else:
        # pendulum: logs_remoteから読み込む
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
    print(f"Loaded {csv_path.name} with {len(df)} rows (seed={seed}, task={task})")
    return df


def smooth_data(data: np.ndarray, method: str, window: int = 10, alpha: float = 0.1, sigma: float = 2.0) -> np.ndarray:
    """データをスムージングする
    
    Args:
        data: 1D array
        method: 'moving_average', 'gaussian', 'ema', 'savitzky_golay', or 'none'
        window: ウィンドウサイズ（moving_average, savitzky_golay用）
        alpha: 平滑化係数（ema用、0-1、小さいほど滑らか）
        sigma: ガウシアンフィルタの標準偏差（gaussian用）
    
    Returns:
        スムージングされたデータ
    """
    if method == 'none' or method is None:
        return data
    
    elif method == 'moving_average':
        # 移動平均
        return pd.Series(data).rolling(window=window, center=True, min_periods=1).mean().values
    
    elif method == 'gaussian':
        # ガウシアンフィルタ
        return gaussian_filter1d(data, sigma=sigma)
    
    elif method == 'ema':
        # 指数移動平均
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    elif method == 'savitzky_golay':
        # Savitzky-Golayフィルタ（より高品質だが遅い）
        window = max(window, 5)  # 最小5
        if window % 2 == 0:  # 奇数にする
            window += 1
        polyorder = min(3, window - 1)  # 多項式次数
        return savgol_filter(data, window_length=window, polyorder=polyorder)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def compute_statistics(dfs: List[pd.DataFrame], smooth_method: str = 'none', 
                       smooth_window: int = 10, smooth_alpha: float = 0.1,
                       smooth_sigma: float = 2.0) -> tuple:
    """複数seedのデータから平均と95%信頼区間を計算"""
    # 全seedのステップを統一
    min_length = min(len(df) for df in dfs)
    aligned_dfs = [df.iloc[:min_length].reset_index(drop=True) for df in dfs]
    
    # 報酬を抽出してスムージング
    rewards = np.array([
        smooth_data(df['episode_reward'].values, smooth_method, 
                   smooth_window, smooth_alpha, smooth_sigma)
        for df in aligned_dfs
    ])
    steps = aligned_dfs[0]['step'].values
    
    # 統計量を計算
    means = np.mean(rewards, axis=0)
    stds = np.std(rewards, axis=0)
    ci_lows = np.percentile(rewards, 2.5, axis=0)
    ci_highs = np.percentile(rewards, 97.5, axis=0)
    
    return steps, means, stds, ci_lows, ci_highs


def plot_learning_curves(data: Dict[str, List[pd.DataFrame]], output_path: Path,
                         smooth_method: str = 'none', smooth_window: int = 10,
                         smooth_alpha: float = 0.1, smooth_sigma: float = 2.0,
                         task: str = "pendulum"):
    """学習曲線をプロット"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model in ["baseline", "dr", "c", "o"]:
        if model not in data or len(data[model]) == 0:
            print(f"Skipping {model}: no data")
            continue
        
        steps, means, stds, ci_lows, ci_highs = compute_statistics(
            data[model], smooth_method, smooth_window, smooth_alpha, smooth_sigma
        )
        
        # メインライン
        ax.plot(steps, means, label=LABELS[model], color=COLORS[model], 
                linewidth=2.5, alpha=0.9)
        
        # 信頼区間
        ax.fill_between(steps, ci_lows, ci_highs, color=COLORS[model], alpha=0.2)
    
    ax.set_xlabel('Environment Steps', fontsize=13, fontweight='bold')
    ax.set_ylabel('Episode Return', fontsize=13, fontweight='bold')
    
    # タイトルにタスクとスムージング情報を追加
    task_title = TASK_TITLES.get(task, task)
    title = f'Learning Curves: Sample Efficiency Comparison ({task_title})'
    if smooth_method != 'none':
        title += f' (Smoothing: {smooth_method})'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
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


def parse_args():
    parser = argparse.ArgumentParser(description='Plot learning curves with optional smoothing')
    
    parser.add_argument('--task', type=str, default='pendulum',
                       choices=['pendulum', 'cheetah', 'walker_actuator'],
                       help='Task to plot (default: pendulum)')
    parser.add_argument('--logs-dir', type=str, default=None,
                       help='Directory containing logs (default: logs_remote for pendulum, logs for walker)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: eval_curves_{task}.png)')
    parser.add_argument('--smooth', type=str, default='none',
                       choices=['none', 'moving_average', 'gaussian', 'ema', 'savitzky_golay'],
                       help='Smoothing method (default: none)')
    parser.add_argument('--window', type=int, default=10,
                       help='Window size for moving_average or savitzky_golay (default: 10)')
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Alpha for exponential moving average, 0-1 (default: 0.1, smaller = smoother)')
    parser.add_argument('--sigma', type=float, default=2.0,
                       help='Sigma for gaussian filter (default: 2.0)')
    parser.add_argument('--num-seeds', type=int, default=5,
                       help='Number of seeds to load (default: 5)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 出力パスのデフォルト設定
    if args.output is None:
        output_path = Path(f"eval_curves_{args.task}.png")
    else:
        output_path = Path(args.output)
    
    if args.logs_dir is None:
        if args.task == "pendulum":
            logs_dir = Path("logs_remote")
        else:
            logs_dir = Path("logs")
    else:
        logs_dir = Path(args.logs_dir)
    
    # pendulum/walkerの場合のみlogs_dirの存在確認
    if args.task in {"pendulum", "walker_actuator"} and not logs_dir.exists():
        print(f"Error: {logs_dir} not found!")
        return
    
    # cheetahの場合はartifactsディレクトリの確認
    if args.task == "cheetah":
        artifacts_dir = Path("artifacts")
        if not artifacts_dir.exists():
            print(f"Error: {artifacts_dir} not found!")
            return
    
    print(f"Configuration:")
    print(f"  Task: {args.task}")
    if args.task in {"pendulum", "walker_actuator"}:
        print(f"  Logs dir: {logs_dir}")
    else:
        print(f"  Artifacts dir: artifacts/")
    print(f"  Output: {output_path}")
    print(f"  Smoothing: {args.smooth}")
    if args.smooth == 'moving_average' or args.smooth == 'savitzky_golay':
        print(f"  Window: {args.window}")
    elif args.smooth == 'ema':
        print(f"  Alpha: {args.alpha}")
    elif args.smooth == 'gaussian':
        print(f"  Sigma: {args.sigma}")
    print()
    
    # データを読み込む
    data = {
        "baseline": [],
        "dr": [],
        "c": [],
        "o": []
    }
    
    for seed in range(args.num_seeds):
        for model in ["baseline", "dr", "c", "o"]:
            df = load_eval_csv(model, seed, logs_dir, task=args.task)
            if df is not None:
                data[model].append(df)
    
    # データが存在するか確認
    if all(len(v) == 0 for v in data.values()):
        print("Error: No data loaded!")
        return
    
    # プロット
    plot_learning_curves(data, output_path, args.smooth, args.window, args.alpha, args.sigma, task=args.task)
    
    # サマリー出力
    print_summary(data)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()

