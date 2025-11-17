"""
Model B (DR), Model C, Model O の結果分析

使用方法:
    python analyze_results.py
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


def smooth_curve(data, weight=0.9):
    """指数移動平均でスムージング"""
    smoothed = []
    last = data[0] if len(data) > 0 else 0
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def load_training_data(log_dirs, model_names, seeds=[0]):
    """
    学習データをロード
    
    Args:
        log_dirs: ログディレクトリのリスト
        model_names: モデル名のリスト
        seeds: シードのリスト
    
    Returns:
        results: {model_name: {'steps': [...], 'rewards': [...]}}
    """
    results = {}
    
    for model_name, log_dir in zip(model_names, log_dirs):
        all_rewards = []
        all_steps = None
        
        for seed in seeds:
            csv_path = Path(log_dir) / str(seed) / 'train.csv'
            
            if not csv_path.exists():
                print(f"Warning: {csv_path} not found")
                continue
            
            df = pd.read_csv(csv_path)
            
            # episode_reward列を取得
            if 'episode_reward' in df.columns:
                rewards = df['episode_reward'].values
                steps = df['step'].values
                
                # NaNを除去
                mask = ~np.isnan(rewards)
                rewards = rewards[mask]
                steps = steps[mask]
                
                all_rewards.append(rewards)
                if all_steps is None:
                    all_steps = steps
        
        if len(all_rewards) > 0:
            # 平均と標準偏差を計算
            # 長さを揃える
            min_len = min(len(r) for r in all_rewards)
            all_rewards = [r[:min_len] for r in all_rewards]
            all_steps = all_steps[:min_len]
            
            rewards_array = np.array(all_rewards)
            mean_rewards = rewards_array.mean(axis=0)
            std_rewards = rewards_array.std(axis=0)
            
            results[model_name] = {
                'steps': all_steps,
                'mean': mean_rewards,
                'std': std_rewards,
                'raw': rewards_array,
            }
            
            print(f"✓ Loaded {model_name}: {len(all_rewards)} seeds, {len(all_steps)} steps")
        else:
            print(f"✗ No data for {model_name}")
    
    return results


def plot_learning_curves(results, save_path='learning_curves.png', smooth_weight=0.95):
    """学習曲線をプロット"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'Model B (DR)': '#1f77b4', 'Model C': '#ff7f0e', 'Model O': '#2ca02c'}
    
    for model_name, data in results.items():
        steps = data['steps']
        mean = data['mean']
        std = data['std']
        
        # スムージング
        mean_smooth = smooth_curve(mean, weight=smooth_weight)
        
        color = colors.get(model_name, None)
        
        # 平均
        ax.plot(steps, mean_smooth, label=model_name, linewidth=2.5, color=color)
        
        # 標準偏差
        ax.fill_between(steps, mean_smooth - std, mean_smooth + std, alpha=0.2, color=color)
    
    ax.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=14, fontweight='bold')
    ax.set_title('Learning Curves: Model B (DR) vs Model C vs Model O', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_final_performance(results, save_path='final_performance.png'):
    """最終性能の比較（箱ひげ図）"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = []
    final_rewards = []
    
    for model_name, data in results.items():
        model_names.append(model_name)
        # 最後の10%のデータの平均を使用
        last_10_percent = int(len(data['mean']) * 0.9)
        final_reward = data['mean'][last_10_percent:].mean()
        final_rewards.append(final_reward)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(model_names, final_rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # 値をバーの上に表示
    for bar, reward in zip(bars, final_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.1f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Average Episode Reward (Last 10%)', fontsize=14, fontweight='bold')
    ax.set_title('Final Performance Comparison', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def print_statistics(results):
    """統計情報を表示"""
    print("\n" + "="*70)
    print("統計情報")
    print("="*70)
    
    for model_name, data in results.items():
        last_10_percent = int(len(data['mean']) * 0.9)
        final_mean = data['mean'][last_10_percent:].mean()
        final_std = data['std'][last_10_percent:].mean()
        max_reward = data['mean'].max()
        
        print(f"\n{model_name}:")
        print(f"  最終性能 (最後10%の平均): {final_mean:.2f} ± {final_std:.2f}")
        print(f"  最大報酬: {max_reward:.2f}")
        print(f"  総ステップ数: {len(data['steps'])}")
    
    # 性能比較
    print("\n" + "="*70)
    print("性能比較")
    print("="*70)
    
    if 'Model B (DR)' in results and 'Model C' in results:
        dr_final = results['Model B (DR)']['mean'][-int(len(results['Model B (DR)']['mean'])*0.1):].mean()
        c_final = results['Model C']['mean'][-int(len(results['Model C']['mean'])*0.1):].mean()
        improvement_c = ((c_final - dr_final) / dr_final) * 100
        print(f"Model C vs Model B (DR): +{improvement_c:.1f}% ({c_final:.1f} vs {dr_final:.1f})")
    
    if 'Model C' in results and 'Model O' in results:
        c_final = results['Model C']['mean'][-int(len(results['Model C']['mean'])*0.1):].mean()
        o_final = results['Model O']['mean'][-int(len(results['Model O']['mean'])*0.1):].mean()
        gap = o_final - c_final
        ratio = (c_final / o_final) * 100
        print(f"Model C vs Model O: {ratio:.1f}% of oracle ({c_final:.1f} vs {o_final:.1f}, gap: {gap:.1f})")
    
    if 'Model B (DR)' in results and 'Model O' in results:
        dr_final = results['Model B (DR)']['mean'][-int(len(results['Model B (DR)']['mean'])*0.1):].mean()
        o_final = results['Model O']['mean'][-int(len(results['Model O']['mean'])*0.1):].mean()
        potential = o_final - dr_final
        print(f"Theoretical potential: {potential:.1f} ({o_final:.1f} - {dr_final:.1f})")


def main():
    """メイン関数"""
    print("="*70)
    print("Model B (DR) vs Model C vs Model O - 結果分析")
    print("="*70)
    
    # ログディレクトリの設定
    base_dir = Path("tdmpc2/logs")
    
    log_dirs = [
        base_dir / "pendulum-swingup-randomized",  # Model B (DR)
        base_dir / "pendulum-swingup-randomized_model_c",  # Model C
        base_dir / "pendulum-swingup-randomized_oracle",  # Model O
    ]
    
    model_names = [
        "Model B (DR)",
        "Model C",
        "Model O",
    ]
    
    # シードの設定（複数シードがある場合）
    seeds = [0]  # 必要に応じて [0, 1, 2] など
    
    # データロード
    results = load_training_data(log_dirs, model_names, seeds)
    
    if len(results) == 0:
        print("エラー: データが見つかりませんでした")
        return
    
    # プロット
    plot_learning_curves(results, save_path='learning_curves.png')
    plot_final_performance(results, save_path='final_performance.png')
    
    # 統計情報
    print_statistics(results)
    
    print("\n" + "="*70)
    print("✓ 分析完了！")
    print("="*70)
    print("生成されたファイル:")
    print("  - learning_curves.png: 学習曲線")
    print("  - final_performance.png: 最終性能比較")


if __name__ == '__main__':
    main()

