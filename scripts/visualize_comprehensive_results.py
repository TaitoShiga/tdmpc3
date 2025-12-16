#!/usr/bin/env python
"""seed別結果CSVの包括的可視化スクリプト"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import List, Tuple

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# カラーパレット
COLORS = {
    'baseline': '#1f77b4',
    'dr': '#ff7f0e',
    'c': '#2ca02c',
    'o': '#d62728'
}

MODEL_NAMES = {
    'baseline': 'Baseline',
    'dr': 'DR',
    'c': 'Model C',
    'o': 'Oracle'
}


def load_all_results(seeds: List[int]) -> pd.DataFrame:
    """全seedのCSVをロードして統合"""
    dfs = []
    for seed in seeds:
        csv_path = Path(f"results_seed{seed}.csv")
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError("No result CSVs found")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} episodes from {len(dfs)} seeds")
    return combined


def compute_iqm(values: np.ndarray) -> float:
    """IQM (Interquartile Mean) を計算"""
    if len(values) == 0:
        return np.nan
    q1, q3 = np.percentile(values, [25, 75])
    iqm_values = values[(values >= q1) & (values <= q3)]
    return np.mean(iqm_values) if len(iqm_values) > 0 else np.mean(values)


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    """ブートストラップで信頼区間を計算"""
    if len(values) < 2:
        return np.nan, np.nan
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(compute_iqm(sample))
    
    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    
    return lower, upper


def compute_statistics(df: pd.DataFrame, train_min: float = 0.5, train_max: float = 2.5) -> pd.DataFrame:
    """モデル×パラメータごとの統計量を計算"""
    stats_list = []
    
    for model in df['model'].unique():
        for param in sorted(df['param'].unique()):
            subset = df[(df['model'] == model) & (df['param'] == param)]
            returns = subset['return'].values
            
            if len(returns) == 0:
                continue
            
            iqm = compute_iqm(returns)
            ci_lower, ci_upper = bootstrap_ci(returns)
            
            # OOD判定
            is_ood = param < train_min or param > train_max
            if param < train_min:
                ood_side = 'light'
            elif param > train_max:
                ood_side = 'heavy'
            else:
                ood_side = 'in-distribution'
            
            stats_list.append({
                'model': model,
                'param': param,
                'iqm': iqm,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'mean': np.mean(returns),
                'std': np.std(returns),
                'median': np.median(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'n_episodes': len(returns),
                'is_ood': is_ood,
                'ood_side': ood_side
            })
    
    return pd.DataFrame(stats_list)


def plot_performance_curves(stats_df: pd.DataFrame, output_path: Path, train_min: float, train_max: float):
    """メイン性能曲線プロット"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 訓練範囲を背景表示
    ax.axvspan(train_min, train_max, alpha=0.15, color='green', 
               label=f'Training Range [{train_min}, {train_max}]', zorder=1)
    
    # 各モデルをプロット
    model_order = ['baseline', 'dr', 'c', 'o']
    for model in model_order:
        if model not in stats_df['model'].values:
            continue
        
        model_data = stats_df[stats_df['model'] == model].sort_values('param')
        
        params = model_data['param'].values
        iqms = model_data['iqm'].values
        ci_lower = model_data['ci_lower'].values
        ci_upper = model_data['ci_upper'].values
        
        # メインライン
        ax.plot(params, iqms, 
                marker='o', markersize=10, linewidth=3,
                color=COLORS[model],
                label=MODEL_NAMES[model],
                zorder=10)
        
        # 信頼区間
        ax.fill_between(params, ci_lower, ci_upper,
                        alpha=0.25, color=COLORS[model],
                        zorder=5)
    
    # OOD境界線
    ax.axvline(train_min, color='red', linestyle='--', linewidth=2.5, 
               alpha=0.7, label='OOD Boundary', zorder=8)
    ax.axvline(train_max, color='red', linestyle='--', linewidth=2.5, alpha=0.7, zorder=8)
    
    # 装飾
    ax.set_xlabel('Mass Multiplier', fontsize=16, fontweight='bold')
    ax.set_ylabel('Episode Return (IQM)', fontsize=16, fontweight='bold')
    ax.set_title('Performance Across Mass Range (In-Distribution + OOD)', 
                 fontsize=18, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
    ax.legend(fontsize=13, loc='best', framealpha=0.95, edgecolor='black', fancybox=True)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance curves: {output_path}")
    plt.close()


def plot_ood_comparison(stats_df: pd.DataFrame, output_path: Path, train_min: float, train_max: float):
    """In-Distribution vs OOD比較"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    categories = [
        ('In-Distribution', 'in-distribution'),
        ('OOD Light', 'light'),
        ('OOD Heavy', 'heavy')
    ]
    
    model_order = ['baseline', 'dr', 'c', 'o']
    x_pos = np.arange(len(model_order))
    bar_width = 0.6
    
    for ax, (title, ood_side) in zip(axes, categories):
        category_data = stats_df[stats_df['ood_side'] == ood_side]
        
        if len(category_data) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue
        
        # モデルごとの平均IQM
        model_iqms = []
        model_cis = []
        for model in model_order:
            model_subset = category_data[category_data['model'] == model]
            if len(model_subset) > 0:
                avg_iqm = model_subset['iqm'].mean()
                # CI範囲の平均幅
                ci_width = (model_subset['ci_upper'] - model_subset['ci_lower']).mean() / 2
                model_iqms.append(avg_iqm)
                model_cis.append(ci_width)
            else:
                model_iqms.append(0)
                model_cis.append(0)
        
        # バープロット
        bars = ax.bar(x_pos, model_iqms, bar_width, 
                      color=[COLORS[m] for m in model_order],
                      alpha=0.85, edgecolor='black', linewidth=1.5,
                      yerr=model_cis, capsize=5)
        
        # 値表示
        for i, (bar, iqm) in enumerate(zip(bars, model_iqms)):
            if iqm > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{iqm:.0f}',
                       ha='center', va='bottom',
                       fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Average IQM', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([MODEL_NAMES[m] for m in model_order], rotation=0)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(model_iqms) * 1.15 if max(model_iqms) > 0 else 1000)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ OOD comparison: {output_path}")
    plt.close()


def plot_degradation_analysis(stats_df: pd.DataFrame, output_path: Path, train_min: float, train_max: float):
    """性能劣化分析"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    model_order = ['baseline', 'dr', 'c', 'o']
    
    # 各モデルのID性能を取得
    id_performance = {}
    for model in model_order:
        id_data = stats_df[(stats_df['model'] == model) & (stats_df['ood_side'] == 'in-distribution')]
        if len(id_data) > 0:
            id_performance[model] = id_data['iqm'].mean()
        else:
            id_performance[model] = np.nan
    
    # 左: 軽い側OOD劣化
    light_degradations = []
    for model in model_order:
        light_data = stats_df[(stats_df['model'] == model) & (stats_df['ood_side'] == 'light')]
        if len(light_data) > 0 and not np.isnan(id_performance[model]):
            ood_perf = light_data['iqm'].mean()
            degradation = (id_performance[model] - ood_perf) / id_performance[model] * 100
            light_degradations.append(degradation)
        else:
            light_degradations.append(0)
    
    bars1 = ax1.bar(np.arange(len(model_order)), light_degradations,
                    color=[COLORS[m] for m in model_order],
                    alpha=0.85, edgecolor='black', linewidth=1.5)
    
    for bar, deg in zip(bars1, light_degradations):
        if deg != 0:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{deg:.1f}%',
                    ha='center', va='bottom' if deg > 0 else 'top',
                    fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Performance Degradation (%)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Light Side OOD (mass < {train_min})', fontsize=14, fontweight='bold')
    ax1.set_xticks(np.arange(len(model_order)))
    ax1.set_xticklabels([MODEL_NAMES[m] for m in model_order])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(0, color='black', linewidth=1.5)
    
    # 右: 重い側OOD劣化
    heavy_degradations = []
    for model in model_order:
        heavy_data = stats_df[(stats_df['model'] == model) & (stats_df['ood_side'] == 'heavy')]
        if len(heavy_data) > 0 and not np.isnan(id_performance[model]):
            ood_perf = heavy_data['iqm'].mean()
            degradation = (id_performance[model] - ood_perf) / id_performance[model] * 100
            heavy_degradations.append(degradation)
        else:
            heavy_degradations.append(0)
    
    bars2 = ax2.bar(np.arange(len(model_order)), heavy_degradations,
                    color=[COLORS[m] for m in model_order],
                    alpha=0.85, edgecolor='black', linewidth=1.5)
    
    for bar, deg in zip(bars2, heavy_degradations):
        if deg != 0:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{deg:.1f}%',
                    ha='center', va='bottom' if deg > 0 else 'top',
                    fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Performance Degradation (%)', fontsize=13, fontweight='bold')
    ax2.set_title(f'Heavy Side OOD (mass > {train_max})', fontsize=14, fontweight='bold')
    ax2.set_xticks(np.arange(len(model_order)))
    ax2.set_xticklabels([MODEL_NAMES[m] for m in model_order])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(0, color='black', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Degradation analysis: {output_path}")
    plt.close()


def plot_heatmap(stats_df: pd.DataFrame, output_path: Path):
    """モデル×質量のヒートマップ"""
    pivot = stats_df.pivot_table(values='iqm', index='model', columns='param', aggfunc='mean')
    
    # モデルの順序
    model_order = ['baseline', 'dr', 'c', 'o']
    pivot = pivot.reindex([m for m in model_order if m in pivot.index])
    pivot.index = [MODEL_NAMES[m] for m in pivot.index]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn',
                cbar_kws={'label': 'Episode Return (IQM)'},
                linewidths=1, linecolor='gray',
                ax=ax, vmin=0, vmax=1000,
                annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_xlabel('Mass Multiplier', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Performance Heatmap: Model × Mass', fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap: {output_path}")
    plt.close()


def plot_seed_variability(df: pd.DataFrame, output_path: Path):
    """Seed間の変動を可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    model_order = ['baseline', 'dr', 'c', 'o']
    
    for ax, model in zip(axes, model_order):
        model_data = df[df['model'] == model]
        
        # seed別にIQMを計算
        seed_iqms = {}
        for seed in sorted(model_data['seed'].unique()):
            for param in sorted(model_data['param'].unique()):
                subset = model_data[(model_data['seed'] == seed) & (model_data['param'] == param)]
                if len(subset) > 0:
                    iqm = compute_iqm(subset['return'].values)
                    if seed not in seed_iqms:
                        seed_iqms[seed] = {'params': [], 'iqms': []}
                    seed_iqms[seed]['params'].append(param)
                    seed_iqms[seed]['iqms'].append(iqm)
        
        # 各seedをプロット
        for seed, data in seed_iqms.items():
            ax.plot(data['params'], data['iqms'], 
                   marker='o', alpha=0.4, linewidth=1.5,
                   label=f'Seed {seed}')
        
        # 平均を太線で
        stats = compute_statistics(model_data)
        ax.plot(stats['param'], stats['iqm'], 
               color='black', linewidth=3, marker='D', markersize=8,
               label='Average', zorder=10)
        
        ax.set_xlabel('Mass Multiplier', fontsize=11, fontweight='bold')
        ax.set_ylabel('Episode Return (IQM)', fontsize=11, fontweight='bold')
        ax.set_title(MODEL_NAMES[model], fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Seed variability: {output_path}")
    plt.close()


def plot_return_distributions(df: pd.DataFrame, output_path: Path, train_min: float, train_max: float):
    """リターン分布のバイオリンプロット"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    params = sorted(df['param'].unique())
    model_order = ['baseline', 'dr', 'c', 'o']
    
    for ax, param in zip(axes.flatten(), params):
        param_data = df[df['param'] == param]
        
        data_list = []
        labels = []
        colors = []
        
        for model in model_order:
            model_subset = param_data[param_data['model'] == model]
            if len(model_subset) > 0:
                data_list.append(model_subset['return'].values)
                labels.append(MODEL_NAMES[model])
                colors.append(COLORS[model])
        
        if data_list:
            parts = ax.violinplot(data_list, positions=range(len(data_list)),
                                  widths=0.7, showmeans=True, showmedians=True)
            
            # 色付け
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=0)
            
            # OODマーカー
            ood_status = "OOD" if (param < train_min or param > train_max) else "ID"
            ax.set_title(f'Mass = {param} ({ood_status})', fontsize=12, fontweight='bold')
            ax.set_ylabel('Episode Return', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Return distributions: {output_path}")
    plt.close()


def generate_summary_table(stats_df: pd.DataFrame, output_path: Path, train_min: float, train_max: float):
    """サマリーテーブル生成"""
    summary = []
    
    model_order = ['baseline', 'dr', 'c', 'o']
    
    for model in model_order:
        model_data = stats_df[stats_df['model'] == model]
        
        # In-Distribution
        id_data = model_data[model_data['ood_side'] == 'in-distribution']
        id_mean = id_data['iqm'].mean() if len(id_data) > 0 else np.nan
        
        # OOD Light
        light_data = model_data[model_data['ood_side'] == 'light']
        light_mean = light_data['iqm'].mean() if len(light_data) > 0 else np.nan
        
        # OOD Heavy
        heavy_data = model_data[model_data['ood_side'] == 'heavy']
        heavy_mean = heavy_data['iqm'].mean() if len(heavy_data) > 0 else np.nan
        
        # Overall
        overall_mean = model_data['iqm'].mean()
        
        # 劣化率
        light_deg = ((id_mean - light_mean) / id_mean * 100) if not np.isnan(id_mean) and not np.isnan(light_mean) else np.nan
        heavy_deg = ((id_mean - heavy_mean) / id_mean * 100) if not np.isnan(id_mean) and not np.isnan(heavy_mean) else np.nan
        
        summary.append({
            'Model': MODEL_NAMES[model],
            f'In-Distribution [{train_min}-{train_max}]': f'{id_mean:.1f}' if not np.isnan(id_mean) else 'N/A',
            f'OOD Light (<{train_min})': f'{light_mean:.1f}' if not np.isnan(light_mean) else 'N/A',
            f'OOD Heavy (>{train_max})': f'{heavy_mean:.1f}' if not np.isnan(heavy_mean) else 'N/A',
            'Overall': f'{overall_mean:.1f}',
            'Light Degradation (%)': f'{light_deg:.1f}' if not np.isnan(light_deg) else 'N/A',
            'Heavy Degradation (%)': f'{heavy_deg:.1f}' if not np.isnan(heavy_deg) else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_path, index=False)
    
    print("\n" + "="*100)
    print("Summary Table: Performance Across All Conditions")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100 + "\n")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Comprehensive visualization of results")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Seeds to include"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures"
    )
    parser.add_argument(
        "--train-min",
        type=float,
        default=0.5,
        help="Training range minimum"
    )
    parser.add_argument(
        "--train-max",
        type=float,
        default=2.5,
        help="Training range maximum"
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Comprehensive Results Visualization")
    print("="*70)
    print(f"Seeds: {args.seeds}")
    print(f"Training range: [{args.train_min}, {args.train_max}]")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # データロード
    print("Loading data...")
    df = load_all_results(args.seeds)
    
    # 統計計算
    print("Computing statistics...")
    stats_df = compute_statistics(df, args.train_min, args.train_max)
    
    print(f"✓ {len(stats_df)} model-param combinations\n")
    
    # 可視化生成
    print("Generating visualizations...\n")
    
    # 1. メイン性能曲線
    plot_performance_curves(
        stats_df, 
        args.output_dir / "performance_curves.png",
        args.train_min, 
        args.train_max
    )
    
    # 2. OOD比較
    plot_ood_comparison(
        stats_df,
        args.output_dir / "ood_comparison.png",
        args.train_min,
        args.train_max
    )
    
    # 3. 劣化分析
    plot_degradation_analysis(
        stats_df,
        args.output_dir / "degradation_analysis.png",
        args.train_min,
        args.train_max
    )
    
    # 4. ヒートマップ
    plot_heatmap(
        stats_df,
        args.output_dir / "heatmap.png"
    )
    
    # 5. Seed変動
    plot_seed_variability(
        df,
        args.output_dir / "seed_variability.png"
    )
    
    # 6. リターン分布
    plot_return_distributions(
        df,
        args.output_dir / "return_distributions.png",
        args.train_min,
        args.train_max
    )
    
    # 7. サマリーテーブル
    summary_df = generate_summary_table(
        stats_df,
        args.output_dir / "summary_table.csv",
        args.train_min,
        args.train_max
    )
    
    print("\n" + "="*70)
    print("✅ All visualizations complete!")
    print("="*70)
    print(f"\nGenerated files in {args.output_dir}/:")
    print("  - performance_curves.png       : Main performance across mass range")
    print("  - ood_comparison.png           : In-Dist vs OOD comparison")
    print("  - degradation_analysis.png     : Performance degradation by OOD side")
    print("  - heatmap.png                  : Model × Mass heatmap")
    print("  - seed_variability.png         : Variability across seeds")
    print("  - return_distributions.png     : Return distribution violin plots")
    print("  - summary_table.csv            : Numerical summary table")
    print()


if __name__ == "__main__":
    main()

