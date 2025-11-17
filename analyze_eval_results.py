"""
評価結果の分析スクリプト（eval.csv ベース）

各モデルの評価結果を比較・可視化します。

使用方法:
    python analyze_eval_results.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Seabornスタイル
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_eval_data(eval_paths, model_names):
    """
    評価データをロード
    
    Args:
        eval_paths: eval.csvのパスのリスト
        model_names: モデル名のリスト
    
    Returns:
        results: {model_name: DataFrame}
    """
    results = {}
    
    for model_name, eval_path in zip(model_names, eval_paths):
        eval_path = Path(eval_path)
        
        if not eval_path.exists():
            print(f"Warning: {eval_path} not found")
            continue
        
        df = pd.read_csv(eval_path)
        
        # NaNを除去
        df = df.dropna()
        
        results[model_name] = df
        print(f"✓ Loaded {model_name}: {len(df)} evaluation points")
    
    return results


def plot_eval_curves(results, save_path='eval_curves.png'):
    """評価曲線をプロット"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {
        'Model B (DR)': '#1f77b4',
        'Model C': '#ff7f0e',
        'Model O': '#2ca02c'
    }
    markers = {
        'Model B (DR)': 'o',
        'Model C': 's',
        'Model O': '^'
    }
    
    for model_name, df in results.items():
        steps = df['step'].values
        rewards = df['episode_reward'].values
        
        color = colors.get(model_name, None)
        marker = markers.get(model_name, 'o')
        
        ax.plot(steps, rewards, label=model_name, linewidth=2.5, 
                marker=marker, markersize=8, color=color, alpha=0.8)
    
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Evaluation Episode Reward', fontsize=14, fontweight='bold')
    ax.set_title('Evaluation Performance: Model B (DR) vs Model C vs Model O', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_final_comparison(results, save_path='final_comparison.png', last_n=3):
    """最終性能の比較（最後のN回の平均）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = []
    final_rewards = []
    final_stds = []
    
    for model_name, df in results.items():
        model_names.append(model_name)
        
        # 最後のN回の評価の平均
        last_rewards = df['episode_reward'].values[-last_n:]
        final_reward = last_rewards.mean()
        final_std = last_rewards.std()
        
        final_rewards.append(final_reward)
        final_stds.append(final_std)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    x_pos = np.arange(len(model_names))
    
    bars = ax.bar(x_pos, final_rewards, yerr=final_stds, 
                   color=colors[:len(model_names)], alpha=0.7, 
                   edgecolor='black', linewidth=2, capsize=10)
    
    # 値をバーの上に表示
    for i, (bar, reward, std) in enumerate(zip(bars, final_rewards, final_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{reward:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel(f'Average Reward (Last {last_n} Evaluations)', fontsize=14, fontweight='bold')
    ax.set_title('Final Performance Comparison', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def print_statistics(results, last_n=3):
    """統計情報を表示"""
    print("\n" + "="*70)
    print("統計情報")
    print("="*70)
    
    final_perfs = {}
    
    for model_name, df in results.items():
        last_rewards = df['episode_reward'].values[-last_n:]
        final_mean = last_rewards.mean()
        final_std = last_rewards.std()
        max_reward = df['episode_reward'].values.max()
        total_steps = df['step'].values[-1]
        
        final_perfs[model_name] = final_mean
        
        print(f"\n{model_name}:")
        print(f"  最終性能 (最後{last_n}回の平均): {final_mean:.2f} ± {final_std:.2f}")
        print(f"  最大報酬: {max_reward:.2f}")
        print(f"  総学習ステップ数: {int(total_steps):,}")
        print(f"  評価回数: {len(df)}")
    
    # 性能比較
    print("\n" + "="*70)
    print("性能比較 (DR < C < O の検証)")
    print("="*70)
    
    if 'Model B (DR)' in final_perfs and 'Model C' in final_perfs:
        dr_final = final_perfs['Model B (DR)']
        c_final = final_perfs['Model C']
        improvement_c = ((c_final - dr_final) / abs(dr_final)) * 100 if dr_final != 0 else 0
        print(f"\nModel C vs Model B (DR):")
        print(f"  差分: {c_final - dr_final:+.2f} ({'+' if improvement_c > 0 else ''}{improvement_c:.1f}%)")
        print(f"  検証: Model C {'>' if c_final > dr_final else '<=' } Model B (DR) ✓" if c_final > dr_final else "  検証: Model C {'>' if c_final > dr_final else '<='} Model B (DR) ✗")
    
    if 'Model C' in final_perfs and 'Model O' in final_perfs:
        c_final = final_perfs['Model C']
        o_final = final_perfs['Model O']
        gap = o_final - c_final
        ratio = (c_final / o_final) * 100 if o_final != 0 else 0
        print(f"\nModel C vs Model O (Oracle):")
        print(f"  達成率: {ratio:.1f}% ({c_final:.2f} / {o_final:.2f})")
        print(f"  Oracleとのギャップ: {gap:.2f}")
        print(f"  検証: Model C {'<' if c_final < o_final else '>='} Model O ✓" if c_final < o_final else "  検証: Model C {'<' if c_final < o_final else '>='} Model O ✗")
    
    if 'Model B (DR)' in final_perfs and 'Model O' in final_perfs:
        dr_final = final_perfs['Model B (DR)']
        o_final = final_perfs['Model O']
        potential = o_final - dr_final
        print(f"\nTheoretical Potential (Oracle上限):")
        print(f"  ポテンシャル: {potential:.2f} ({o_final:.2f} - {dr_final:.2f})")
    
    # 最終検証
    if len(final_perfs) == 3:
        dr = final_perfs.get('Model B (DR)', 0)
        c = final_perfs.get('Model C', 0)
        o = final_perfs.get('Model O', 0)
        
        verification = dr < c < o
        print(f"\n" + "="*70)
        print(f"✓✓✓ 最終検証: DR < C < O")
        print(f"="*70)
        print(f"  Model B (DR): {dr:.2f}")
        print(f"  Model C:      {c:.2f}  {'✓' if c > dr else '✗'}")
        print(f"  Model O:      {o:.2f}  {'✓' if o > c else '✗'}")
        print(f"\n  結果: {'✓ 成功！ DR < C < O が確認されました' if verification else '✗ 期待された関係が確認されませんでした'}")


def main():
    """メイン関数"""
    print("="*70)
    print("Model B (DR) vs Model C vs Model O - 評価結果分析")
    print("="*70)
    
    # 評価ファイルのパス
    eval_paths = [
        "logs/pendulum-swingup-randomized/0/dr/eval.csv",           # Model B (DR)
        "tdmpc2/logs/pendulum-swingup-randomized/0/default/eval.csv",   # Model C（仮）
        "tdmpc2/logs/pendulum-swingup-randomized/0/oracle_100k/eval.csv",   # Model O
    ]
    
    model_names = [
        "Model B (DR)",
        "Model C",
        "Model O",
    ]
    
    # データロード
    results = load_eval_data(eval_paths, model_names)
    
    if len(results) == 0:
        print("\nエラー: データが見つかりませんでした")
        print("\n利用可能なログディレクトリを確認してください:")
        print("  logs/pendulum-swingup-randomized/0/")
        return
    
    # プロット
    plot_eval_curves(results, save_path='eval_curves.png')
    plot_final_comparison(results, save_path='final_comparison.png', last_n=3)
    
    # 統計情報
    print_statistics(results, last_n=3)
    
    print("\n" + "="*70)
    print("✓ 分析完了！")
    print("="*70)
    print("生成されたファイル:")
    print("  - eval_curves.png: 評価曲線")
    print("  - final_comparison.png: 最終性能比較")


if __name__ == '__main__':
    main()

