#!/usr/bin/env python
"""両側OOD評価結果の可視化スクリプト"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# カラーパレット
COLORS = {
    'baseline': '#1f77b4',
    'dr': '#ff7f0e',
    'model_c': '#2ca02c',
    'oracle': '#d62728'
}

MODEL_NAMES = {
    'baseline': 'Baseline',
    'dr': 'DR',
    'model_c': 'Model C',
    'oracle': 'Oracle'
}


def load_evaluation_results(output_dir: Path) -> Dict:
    """評価結果をロード"""
    results = {
        'baseline': {},
        'dr': {},
        'model_c': {},
        'oracle': {}
    }
    
    # 各seedのディレクトリをスキャン
    for seed_dir in sorted(output_dir.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        
        # 最新のタイムスタンプディレクトリを取得
        timestamp_dirs = sorted(seed_dir.glob("*"))
        if not timestamp_dirs:
            continue
        
        latest_dir = timestamp_dirs[-1]
        results_json = latest_dir / "results.json"
        
        if not results_json.exists():
            print(f"Warning: {results_json} not found, skipping")
            continue
        
        with open(results_json, 'r') as f:
            data = json.load(f)
        
        # Mass別に結果を集約
        for mass_key, mass_results in data['results_by_mass'].items():
            mass = float(mass_key)
            episodes = mass_results['episodes']
            
            # 各モデルタイプを推定（簡易版）
            # 実際のモデルタイプは評価結果に含まれていることを想定
            model_type = data.get('model_type', 'baseline')  # デフォルト
            
            if mass not in results[model_type]:
                results[model_type][mass] = []
            
            # エピソード報酬を追加
            returns = [ep['episode_return'] for ep in episodes]
            results[model_type][mass].extend(returns)
    
    return results


def compute_iqm(values: List[float]) -> float:
    """IQM (Interquartile Mean) を計算"""
    if not values:
        return np.nan
    arr = np.array(values)
    q1, q3 = np.percentile(arr, [25, 75])
    iqm_values = arr[(arr >= q1) & (arr <= q3)]
    return np.mean(iqm_values) if len(iqm_values) > 0 else np.mean(arr)


def bootstrap_ci(values: List[float], n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    """ブートストラップで信頼区間を計算"""
    if not values or len(values) < 2:
        return np.nan, np.nan
    
    bootstrap_means = []
    values_array = np.array(values)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(values_array, size=len(values_array), replace=True)
        bootstrap_means.append(compute_iqm(sample))
    
    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    
    return lower, upper


def prepare_plot_data(results: Dict, train_min: float = 0.5, train_max: float = 2.5) -> pd.DataFrame:
    """プロット用のDataFrameを準備"""
    plot_data = []
    
    for model_type, mass_data in results.items():
        for mass, returns in mass_data.items():
            if not returns:
                continue
            
            iqm = compute_iqm(returns)
            lower, upper = bootstrap_ci(returns)
            
            # OODかどうかを判定
            is_ood = mass < train_min or mass > train_max
            ood_side = 'light' if mass < train_min else ('heavy' if mass > train_max else 'in-distribution')
            
            plot_data.append({
                'model': model_type,
                'mass': mass,
                'iqm': iqm,
                'ci_lower': lower,
                'ci_upper': upper,
                'is_ood': is_ood,
                'ood_side': ood_side,
                'n_episodes': len(returns)
            })
    
    return pd.DataFrame(plot_data)


def plot_ood_performance(df: pd.DataFrame, output_path: Path, train_min: float = 0.5, train_max: float = 2.5):
    """OOD性能をプロット"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 訓練範囲を背景色で表示
    ax.axvspan(train_min, train_max, alpha=0.15, color='green', label='Training Range')
    
    # 各モデルをプロット
    for model_type in ['baseline', 'dr', 'model_c', 'oracle']:
        model_df = df[df['model'] == model_type].sort_values('mass')
        
        if len(model_df) == 0:
            continue
        
        masses = model_df['mass'].values
        iqms = model_df['iqm'].values
        ci_lower = model_df['ci_lower'].values
        ci_upper = model_df['ci_upper'].values
        
        # メインライン
        ax.plot(masses, iqms, 
                marker='o', markersize=8, linewidth=2.5,
                color=COLORS.get(model_type, 'gray'),
                label=MODEL_NAMES.get(model_type, model_type),
                zorder=10)
        
        # 信頼区間
        ax.fill_between(masses, ci_lower, ci_upper,
                        alpha=0.2, color=COLORS.get(model_type, 'gray'),
                        zorder=5)
    
    # OOD境界線
    ax.axvline(train_min, color='red', linestyle='--', linewidth=2, alpha=0.7, label='OOD Boundary')
    ax.axvline(train_max, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # ラベルとタイトル
    ax.set_xlabel('Mass Multiplier', fontsize=14, fontweight='bold')
    ax.set_ylabel('Episode Return (IQM)', fontsize=14, fontweight='bold')
    ax.set_title('Out-of-Distribution Performance: Mass Generalization', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # グリッド
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 凡例
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    
    # x軸の範囲を調整
    all_masses = df['mass'].values
    x_margin = (all_masses.max() - all_masses.min()) * 0.05
    ax.set_xlim(all_masses.min() - x_margin, all_masses.max() + x_margin)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ OOD performance plot saved: {output_path}")
    plt.close()


def plot_ood_degradation(df: pd.DataFrame, output_path: Path, train_min: float = 0.5, train_max: float = 2.5):
    """OOD性能劣化を可視化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 各モデルのID性能を取得（訓練範囲内の平均）
    id_performance = {}
    for model_type in ['baseline', 'dr', 'model_c', 'oracle']:
        model_df = df[(df['model'] == model_type) & (~df['is_ood'])]
        if len(model_df) > 0:
            id_performance[model_type] = model_df['iqm'].mean()
        else:
            id_performance[model_type] = np.nan
    
    # 左パネル: 軽い側OOD
    ood_light_data = []
    for model_type in ['baseline', 'dr', 'model_c', 'oracle']:
        light_df = df[(df['model'] == model_type) & (df['ood_side'] == 'light')]
        if len(light_df) > 0:
            ood_perf = light_df['iqm'].mean()
            id_perf = id_performance[model_type]
            if not np.isnan(id_perf) and id_perf > 0:
                degradation = (id_perf - ood_perf) / id_perf * 100
                ood_light_data.append({
                    'model': MODEL_NAMES.get(model_type, model_type),
                    'degradation': degradation
                })
    
    if ood_light_data:
        light_df_plot = pd.DataFrame(ood_light_data)
        bars1 = ax1.bar(light_df_plot['model'], light_df_plot['degradation'],
                       color=[COLORS.get(m.lower().replace(' ', '_'), 'gray') 
                              for m in light_df_plot['model']],
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Performance Degradation (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Light Side OOD (mass < 0.5)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(0, color='black', linewidth=1)
        
        # 値を表示
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    # 右パネル: 重い側OOD
    ood_heavy_data = []
    for model_type in ['baseline', 'dr', 'model_c', 'oracle']:
        heavy_df = df[(df['model'] == model_type) & (df['ood_side'] == 'heavy')]
        if len(heavy_df) > 0:
            ood_perf = heavy_df['iqm'].mean()
            id_perf = id_performance[model_type]
            if not np.isnan(id_perf) and id_perf > 0:
                degradation = (id_perf - ood_perf) / id_perf * 100
                ood_heavy_data.append({
                    'model': MODEL_NAMES.get(model_type, model_type),
                    'degradation': degradation
                })
    
    if ood_heavy_data:
        heavy_df_plot = pd.DataFrame(ood_heavy_data)
        bars2 = ax2.bar(heavy_df_plot['model'], heavy_df_plot['degradation'],
                       color=[COLORS.get(m.lower().replace(' ', '_'), 'gray') 
                              for m in heavy_df_plot['model']],
                       alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Performance Degradation (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Heavy Side OOD (mass > 2.5)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(0, color='black', linewidth=1)
        
        # 値を表示
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ OOD degradation plot saved: {output_path}")
    plt.close()


def plot_ood_heatmap(df: pd.DataFrame, output_path: Path):
    """質量別・モデル別のヒートマップ"""
    # ピボットテーブル作成
    pivot_data = df.pivot_table(
        values='iqm',
        index='mass',
        columns='model',
        aggfunc='mean'
    )
    
    # モデルの順序を指定
    model_order = ['baseline', 'dr', 'model_c', 'oracle']
    pivot_data = pivot_data[[col for col in model_order if col in pivot_data.columns]]
    pivot_data.columns = [MODEL_NAMES.get(col, col) for col in pivot_data.columns]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ヒートマップ
    sns.heatmap(pivot_data.T, annot=True, fmt='.0f', cmap='RdYlGn',
                cbar_kws={'label': 'Episode Return (IQM)'},
                linewidths=0.5, linecolor='gray',
                ax=ax, vmin=0, vmax=1000)
    
    ax.set_xlabel('Mass Multiplier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Type', fontsize=12, fontweight='bold')
    ax.set_title('Performance Heatmap: Mass × Model', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ OOD heatmap saved: {output_path}")
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_path: Path, train_min: float = 0.5, train_max: float = 2.5):
    """サマリーテーブルを生成"""
    summary = []
    
    for model_type in ['baseline', 'dr', 'model_c', 'oracle']:
        model_df = df[df['model'] == model_type]
        
        if len(model_df) == 0:
            continue
        
        # In-Distribution
        id_df = model_df[~model_df['is_ood']]
        id_mean = id_df['iqm'].mean() if len(id_df) > 0 else np.nan
        
        # OOD Light
        light_df = model_df[model_df['ood_side'] == 'light']
        light_mean = light_df['iqm'].mean() if len(light_df) > 0 else np.nan
        
        # OOD Heavy
        heavy_df = model_df[model_df['ood_side'] == 'heavy']
        heavy_mean = heavy_df['iqm'].mean() if len(heavy_df) > 0 else np.nan
        
        # Overall
        overall_mean = model_df['iqm'].mean()
        
        summary.append({
            'Model': MODEL_NAMES.get(model_type, model_type),
            'In-Distribution': f"{id_mean:.1f}" if not np.isnan(id_mean) else "N/A",
            'OOD Light (<0.5)': f"{light_mean:.1f}" if not np.isnan(light_mean) else "N/A",
            'OOD Heavy (>2.5)': f"{heavy_mean:.1f}" if not np.isnan(heavy_mean) else "N/A",
            'Overall': f"{overall_mean:.1f}"
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_path, index=False)
    print(f"✓ Summary table saved: {output_path}")
    
    # コンソール表示
    print("\n" + "="*70)
    print("Summary: OOD Performance")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="OOD評価結果の可視化")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/ood_eval_bilateral"),
        help="評価結果のディレクトリ"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ood_visualizations"),
        help="可視化結果の出力ディレクトリ"
    )
    parser.add_argument(
        "--train-min",
        type=float,
        default=0.5,
        help="訓練範囲の最小値"
    )
    parser.add_argument(
        "--train-max",
        type=float,
        default=2.5,
        help="訓練範囲の最大値"
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("OOD Evaluation Visualization")
    print("="*70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training range: [{args.train_min}, {args.train_max}]")
    print()
    
    # 結果をロード
    print("Loading evaluation results...")
    results = load_evaluation_results(args.input_dir)
    
    # データフレーム準備
    df = prepare_plot_data(results, args.train_min, args.train_max)
    
    if len(df) == 0:
        print("Error: No data found. Check input directory.")
        return
    
    print(f"Loaded {len(df)} data points")
    print()
    
    # 可視化生成
    print("Generating visualizations...")
    
    # 1. OOD性能曲線
    plot_ood_performance(
        df, 
        args.output_dir / "ood_performance_curves.png",
        args.train_min,
        args.train_max
    )
    
    # 2. OOD性能劣化
    plot_ood_degradation(
        df,
        args.output_dir / "ood_degradation.png",
        args.train_min,
        args.train_max
    )
    
    # 3. ヒートマップ
    plot_ood_heatmap(
        df,
        args.output_dir / "ood_heatmap.png"
    )
    
    # 4. サマリーテーブル
    generate_summary_table(
        df,
        args.output_dir / "ood_summary.csv",
        args.train_min,
        args.train_max
    )
    
    print()
    print("="*70)
    print("✅ Visualization complete!")
    print("="*70)
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()

