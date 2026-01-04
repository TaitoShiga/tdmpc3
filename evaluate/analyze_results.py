#!/usr/bin/env python
"""
強化学習実験の統計解析スクリプト

results.csv を読み込み、以下を実行：
- IQM (Interquartile Mean) 計算
- ブートストラップによる95% CI推定
- モデル間のペアワイズ比較
- 可視化（棒グラフ、折れ線グラフ）

使用方法:
    python evaluate/analyze_results.py
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # GUIなし環境対応
import matplotlib.pyplot as plt

# 乱数シード固定
np.random.seed(0)

# ブートストラップの回数
BOOTSTRAP_ITERATIONS = 10000


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze evaluation results CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results.csv",
        help="Input results CSV (default: results.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Directory for output figures (default: repo root).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Optional prefix for output figures (default: '').",
    )
    parser.add_argument(
        "--param-label",
        type=str,
        default="Param (mass multiplier)",
        help="X-axis label for param plots.",
    )
    parser.add_argument(
        "--train-min",
        type=float,
        default=None,
        help="Training range minimum (for OOD shading). Default: None (no shading).",
    )
    parser.add_argument(
        "--train-max",
        type=float,
        default=None,
        help="Training range maximum (for OOD shading). Default: None (no shading).",
    )
    return parser.parse_args()


def compute_iqm(values: np.ndarray) -> float:
    """
    IQM (Interquartile Mean) を計算
    
    上位25%と下位25%を除いた中央50%の平均値
    
    Args:
        values: 値の配列
    
    Returns:
        IQM値
    """
    if len(values) == 0:
        return np.nan
    
    sorted_values = np.sort(values)
    q25 = np.percentile(sorted_values, 25)
    q75 = np.percentile(sorted_values, 75)
    
    # 25%〜75%の範囲の値を抽出
    iqr_values = sorted_values[(sorted_values >= q25) & (sorted_values <= q75)]
    
    if len(iqr_values) == 0:
        return np.nan
    
    return np.mean(iqr_values)


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = BOOTSTRAP_ITERATIONS, 
                 statistic_fn=np.mean, alpha: float = 0.05) -> Tuple[float, float]:
    """
    ブートストラップによる信頼区間を計算
    
    Args:
        values: データ配列（seed単位のスコアなど）
        n_bootstrap: ブートストラップ回数
        statistic_fn: 統計量の関数（デフォルトは平均）
        alpha: 有意水準（0.05 = 95% CI）
    
    Returns:
        (ci_low, ci_high): 信頼区間の下限・上限
    """
    if len(values) == 0:
        return np.nan, np.nan
    
    bootstrap_stats = []
    n = len(values)
    
    for _ in range(n_bootstrap):
        # リサンプリング（復元抽出）
        sample = np.random.choice(values, size=n, replace=True)
        stat = statistic_fn(sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    ci_low = np.percentile(bootstrap_stats, alpha/2 * 100)
    ci_high = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    
    return ci_low, ci_high


def load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    """
    results.csvを読み込んで前処理
    
    Args:
        csv_path: CSVファイルのパス
    
    Returns:
        前処理済みDataFrame
    """
    print("="*70)
    print("データ読み込み・前処理")
    print("="*70)
    
    df = pd.read_csv(csv_path)
    
    # modelカラムのフィルタリング
    valid_models = ["baseline", "dr", "c", "o"]
    df = df[df["model"].isin(valid_models)]
    
    # 欠損値を除外
    df = df.dropna()
    
    # 統計情報を表示
    n_combinations = df.groupby(["model", "seed", "param"]).ngroups
    episodes_per_combination = df.groupby(["model", "seed", "param"]).size().mean()
    
    print(f"有効な行数: {len(df)}")
    print(f"(model, seed, param) の組み合わせ数: {n_combinations}")
    print(f"各組み合わせあたりの平均エピソード数: {episodes_per_combination:.1f}")
    print()
    
    return df


def compute_iqm_per_combination(df: pd.DataFrame) -> pd.DataFrame:
    """
    各 (model, seed, param) について IQM を計算
    
    Args:
        df: 元のDataFrame
    
    Returns:
        IQMを含むDataFrame (columns: model, seed, param, iqm_return)
    """
    print("="*70)
    print("IQM計算")
    print("="*70)
    
    iqm_records = []
    
    for (model, seed, param), group in df.groupby(["model", "seed", "param"]):
        returns = group["return"].values
        iqm = compute_iqm(returns)
        
        iqm_records.append({
            "model": model,
            "seed": seed,
            "param": param,
            "iqm_return": iqm
        })
    
    df_iqm = pd.DataFrame(iqm_records)
    
    print(f"IQM計算完了: {len(df_iqm)} 組み合わせ")
    print()
    
    return df_iqm


def aggregate_by_model_param(df_iqm: pd.DataFrame) -> pd.DataFrame:
    """
    model と param ごとに集約（seed をまたいで統計量を計算）
    
    Args:
        df_iqm: IQMを含むDataFrame
    
    Returns:
        集約結果 (columns: model, param, mean, ci_low, ci_high)
    """
    print("="*70)
    print("モデル・パラメータごとの集約 + ブートストラップCI")
    print("="*70)
    
    aggregated = []
    
    for (model, param), group in df_iqm.groupby(["model", "param"]):
        iqm_values = group["iqm_return"].values
        
        mean_val = np.mean(iqm_values)
        ci_low, ci_high = bootstrap_ci(iqm_values, statistic_fn=np.mean)
        
        aggregated.append({
            "model": model,
            "param": param,
            "mean": mean_val,
            "ci_low": ci_low,
            "ci_high": ci_high
        })
    
    df_agg = pd.DataFrame(aggregated)
    
    print(f"集約完了: {len(df_agg)} エントリ")
    print(df_agg.to_string(index=False))
    print()
    
    return df_agg


def aggregate_overall(df_iqm: pd.DataFrame) -> pd.DataFrame:
    """
    param をまたいで全体性能を集約
    
    各 seed について、全 param の IQM を平均したものを「その seed の overall スコア」とする
    
    Args:
        df_iqm: IQMを含むDataFrame
    
    Returns:
        集約結果 (columns: model, mean, ci_low, ci_high)
    """
    print("="*70)
    print("全体性能（param平均）の集約")
    print("="*70)
    
    # 各 (model, seed) について、param をまたいで平均
    df_seed_avg = df_iqm.groupby(["model", "seed"])["iqm_return"].mean().reset_index()
    df_seed_avg.rename(columns={"iqm_return": "overall_iqm"}, inplace=True)
    
    # model ごとに集約
    aggregated = []
    
    for model, group in df_seed_avg.groupby("model"):
        overall_values = group["overall_iqm"].values
        
        mean_val = np.mean(overall_values)
        ci_low, ci_high = bootstrap_ci(overall_values, statistic_fn=np.mean)
        
        aggregated.append({
            "model": model,
            "param": "overall",
            "mean": mean_val,
            "ci_low": ci_low,
            "ci_high": ci_high
        })
    
    df_overall = pd.DataFrame(aggregated)
    
    print("全体性能:")
    print(df_overall.to_string(index=False))
    print()
    
    return df_overall


def pairwise_comparison(df_iqm: pd.DataFrame) -> pd.DataFrame:
    """
    モデル間のペアワイズ差分を計算
    
    Args:
        df_iqm: IQMを含むDataFrame
    
    Returns:
        差分結果 (columns: model_a, model_b, param, mean_diff, ci_low, ci_high, significant)
    """
    print("="*70)
    print("モデル間ペアワイズ比較")
    print("="*70)
    
    models = ["baseline", "dr", "c", "o"]
    pairs = [
        ("baseline", "dr"),
        ("baseline", "c"),
        ("baseline", "o"),
        ("dr", "c"),
        ("dr", "o"),
        ("c", "o"),
    ]
    
    results = []
    
    # param ごとの比較
    for param in df_iqm["param"].unique():
        df_param = df_iqm[df_iqm["param"] == param]
        
        for model_a, model_b in pairs:
            df_a = df_param[df_param["model"] == model_a].set_index("seed")["iqm_return"]
            df_b = df_param[df_param["model"] == model_b].set_index("seed")["iqm_return"]
            
            # 共通のseedのみを使用
            common_seeds = df_a.index.intersection(df_b.index)
            if len(common_seeds) == 0:
                continue
            
            diff = df_b.loc[common_seeds].values - df_a.loc[common_seeds].values
            
            mean_diff = np.mean(diff)
            ci_low, ci_high = bootstrap_ci(diff, statistic_fn=np.mean)
            
            significant = (ci_low > 0) or (ci_high < 0)
            
            results.append({
                "model_a": model_a,
                "model_b": model_b,
                "param": param,
                "mean_diff": mean_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "significant": significant
            })
    
    # overall（param平均）の比較
    df_seed_avg = df_iqm.groupby(["model", "seed"])["iqm_return"].mean().reset_index()
    df_seed_avg.rename(columns={"iqm_return": "overall_iqm"}, inplace=True)
    
    for model_a, model_b in pairs:
        df_a = df_seed_avg[df_seed_avg["model"] == model_a].set_index("seed")["overall_iqm"]
        df_b = df_seed_avg[df_seed_avg["model"] == model_b].set_index("seed")["overall_iqm"]
        
        common_seeds = df_a.index.intersection(df_b.index)
        if len(common_seeds) == 0:
            continue
        
        diff = df_b.loc[common_seeds].values - df_a.loc[common_seeds].values
        
        mean_diff = np.mean(diff)
        ci_low, ci_high = bootstrap_ci(diff, statistic_fn=np.mean)
        
        significant = (ci_low > 0) or (ci_high < 0)
        
        results.append({
            "model_a": model_a,
            "model_b": model_b,
            "param": "overall",
            "mean_diff": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "significant": significant
        })
    
    df_diff = pd.DataFrame(results)
    
    print("ペアワイズ差分:")
    print(df_diff.to_string(index=False))
    print()
    
    return df_diff


def plot_per_param(df_agg: pd.DataFrame, output_path: Path, param_label: str, 
                   train_min: float = None, train_max: float = None):
    """
    param ごとの性能比較グラフを生成（OOD範囲を網掛け表示）
    
    Args:
        df_agg: 集約結果
        output_path: 保存先
        param_label: X軸ラベル
        train_min: 学習範囲の最小値（Noneの場合は網掛けなし）
        train_max: 学習範囲の最大値（Noneの場合は網掛けなし）
    """
    print("="*70)
    print("可視化: param ごとの性能比較")
    print("="*70)
    
    models = ["baseline", "dr", "c", "o"]
    colors = {"baseline": "C0", "dr": "C1", "c": "C2", "o": "C3"}
    labels = {"baseline": "Baseline", "dr": "DR", "c": "Model C", "o": "Oracle"}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # OOD範囲を網掛け表示
    if train_min is not None and train_max is not None:
        all_params = df_agg["param"].unique()
        param_min, param_max = all_params.min(), all_params.max()
        
        # 左側のOOD範囲（軽すぎ）
        if param_min < train_min:
            ax.axvspan(param_min - 0.05, train_min, alpha=0.15, color='gray', label='OOD')
        
        # 右側のOOD範囲（重すぎ）
        if param_max > train_max:
            ax.axvspan(train_max, param_max + 0.05, alpha=0.15, color='gray')
    
    for model in models:
        df_model = df_agg[df_agg["model"] == model].sort_values("param")
        
        params = df_model["param"].values
        means = df_model["mean"].values
        ci_lows = df_model["ci_low"].values
        ci_highs = df_model["ci_high"].values
        
        errors = [means - ci_lows, ci_highs - means]
        
        ax.errorbar(
            params, means, yerr=errors, 
            label=labels[model], marker='o', capsize=5, 
            linewidth=2, markersize=8, color=colors[model]
        )
    
    ax.set_xlabel(param_label, fontsize=12)
    ax.set_ylabel("Mean IQM Return", fontsize=12)
    ax.set_title("Performance vs param (IQM with 95% CI)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"保存: {output_path}")
    plt.show()
    plt.close()
    
    print()


def plot_overall(df_overall: pd.DataFrame, output_path: Path):
    """
    overall 性能比較グラフを生成
    
    Args:
        df_overall: overall集約結果
        output_path: 保存先
    """
    print("="*70)
    print("可視化: overall 性能比較")
    print("="*70)
    
    models = ["baseline", "dr", "c", "o"]
    labels = {"baseline": "Baseline", "dr": "DR", "c": "Model C", "o": "Oracle"}
    colors = {"baseline": "C0", "dr": "C1", "c": "C2", "o": "C3"}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_positions = np.arange(len(models))
    means = []
    errors_low = []
    errors_high = []
    
    for model in models:
        row = df_overall[df_overall["model"] == model].iloc[0]
        means.append(row["mean"])
        errors_low.append(row["mean"] - row["ci_low"])
        errors_high.append(row["ci_high"] - row["mean"])
    
    ax.bar(
        x_positions, means, 
        yerr=[errors_low, errors_high],
        color=[colors[m] for m in models],
        capsize=5, alpha=0.8
    )
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([labels[m] for m in models], fontsize=11)
    ax.set_ylabel("Mean IQM Return (averaged over params)", fontsize=12)
    ax.set_title("Overall performance (IQM averaged over params)", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"保存: {output_path}")
    plt.show()
    plt.close()
    
    print()


def main():
    args = parse_args()
    csv_path = args.input
    output_dir = args.output_dir
    prefix = args.output_prefix
    prefix = f"{prefix}_" if prefix else ""

    if not csv_path.exists():
        print(f"Error: {csv_path} が見つかりません")
        print("まず evaluate/evaluate_all_models.py を実行してください")
        sys.exit(1)
    
    # 1. データ読み込み・前処理
    df = load_and_preprocess(csv_path)
    
    # 2. IQM計算
    df_iqm = compute_iqm_per_combination(df)
    
    # 3. param ごとの集約
    df_agg = aggregate_by_model_param(df_iqm)
    
    # 4. overall 集約
    df_overall = aggregate_overall(df_iqm)
    
    # 5. ペアワイズ比較
    df_diff = pairwise_comparison(df_iqm)
    
    # 6. 可視化
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_per_param(df_agg, output_dir / f"{prefix}fig_per_param.png", args.param_label,
                   train_min=args.train_min, train_max=args.train_max)
    plot_overall(df_overall, output_dir / f"{prefix}fig_overall.png")
    
    # 7. まとめを表示
    print("="*70)
    print("解析完了")
    print("="*70)
    print(f"✓ param ごとの結果: {len(df_agg)} エントリ")
    print(f"✓ overall 結果: {len(df_overall)} エントリ")
    print(f"✓ ペアワイズ比較: {len(df_diff)} ペア")
    print()
    print("生成されたファイル:")
    print(f"  - {output_dir / f'{prefix}fig_per_param.png'}")
    print(f"  - {output_dir / f'{prefix}fig_overall.png'}")
    print()
    
    # 有意な差分を表示
    df_sig = df_diff[df_diff["significant"] == True]
    if len(df_sig) > 0:
        print("統計的に有意な差分:")
        for _, row in df_sig.iterrows():
            print(f"  {row['model_b']} vs {row['model_a']} (param={row['param']}): "
                  f"diff={row['mean_diff']:+.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]")
    else:
        print("統計的に有意な差分は検出されませんでした")


if __name__ == "__main__":
    main()
