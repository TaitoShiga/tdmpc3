"""
Model C の物理パラメータ推定精度の分析スクリプト

GRUがどれだけ正確に質量を推定できているかを多角的に可視化します。

使用方法:
    python analyze_estimation_accuracy.py

生成されるプロット:
    1. estimation_scatter.png - 推定値 vs 真値の散布図
    2. estimation_convergence.png - エピソード内での推定値の収束
    3. estimation_error_dist.png - 推定誤差の分布
    4. estimation_by_mass.png - 質量範囲別の推定精度
    5. estimation_timeline.png - 時系列での推定誤差
    6. estimation_summary.png - 総合ダッシュボード
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tdmpc2'))

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from omegaconf import OmegaConf
import argparse

from envs import make_env
from envs.wrappers.physics_param import wrap_with_physics_param
from tdmpc2_model_c import TDMPC2ModelC
from common.parser import parse_cfg

# 美しいプロット設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description='Model C の物理パラメータ推定精度を分析します。'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Model C (GRU) チェックポイントのパス。未指定の場合は既定候補から探索します。'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='分析に使用するエピソード数'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=500,
        help='各エピソードで収集する最大ステップ数'
    )
    return parser.parse_args()


def resolve_checkpoint_path(user_supplied):
    """チェックポイントのパスを決定"""
    if user_supplied:
        path = Path(user_supplied).expanduser()
        if not path.exists():
            raise FileNotFoundError(f'指定されたチェックポイントが見つかりません: {path}')
        return path
    
    candidates = [
        Path('tdmpc2/logs/pendulum-swingup-randomized/0/default/models/final.pt'),
        Path('logs/pendulum-swingup-randomized/0/default/models/final.pt'),
    ]
    
    for cand in candidates:
        if cand.exists():
            return cand
    
    raise FileNotFoundError(
        'Model C のチェックポイントが自動検出できませんでした。\n'
        ' --checkpoint でパスを指定してください。例:\n'
        '   python analyze_estimation_accuracy.py '
        '--checkpoint tdmpc2/logs/pendulum-swingup-randomized/0/default/models/final.pt'
    )


def load_config_and_agent(checkpoint_path):
    """設定とエージェントをロード"""
    config_path = Path(__file__).parent / 'tdmpc2' / 'config.yaml'
    cfg = OmegaConf.load(config_path)
    
    # 評価用の設定
    cfg.task = 'pendulum-swingup-randomized'
    cfg.obs = 'state'
    cfg.episodic = False
    cfg.seed = 0
    cfg.use_model_c = True
    cfg.use_oracle = False
    cfg.compile = False
    cfg.multitask = False
    
    # ???をデフォルト値に置き換え
    if cfg.get('checkpoint', '???') == '???':
        cfg.checkpoint = None
    if cfg.get('data_dir', '???') == '???':
        cfg.data_dir = None
    if cfg.get('gru_pretrained', '???') == '???':
        cfg.gru_pretrained = None
    
    cfg = parse_cfg(cfg)
    
    # 環境を作成
    env = make_env(cfg)
    env = wrap_with_physics_param(env, cfg)
    
    # エージェントをロード
    checkpoint_data = torch.load(checkpoint_path, map_location=torch.get_default_device(), weights_only=False)
    model_dict = checkpoint_data["model"] if "model" in checkpoint_data else checkpoint_data
    if not any(k.startswith("_physics_estimator") for k in model_dict.keys()):
        raise ValueError(
            f"指定されたチェックポイントはModel C用ではありません: {checkpoint_path}\n"
            "Model C (GRU物理推定器付き) のチェックポイントを指定してください。"
        )
    agent = TDMPC2ModelC(cfg)
    agent.load(checkpoint_path)
    agent.eval()
    
    return cfg, env, agent


def denormalize_mass(normalized_mass, cfg):
    """正規化された質量を元に戻す"""
    if cfg.phys_param_normalization == 'standard':
        # Pendulumの質量範囲: 0.5~2.0, mean=1.25, std=0.433
        mean = 1.25
        std = 0.433
        return normalized_mass * std + mean
    elif cfg.phys_param_normalization == 'minmax':
        # [0, 1] -> [0.5, 2.0]
        return normalized_mass * 1.5 + 0.5
    else:
        return normalized_mass


def collect_estimation_data(agent, env, cfg, num_episodes=20, max_steps=500):
    """
    推定データを収集
    
    Returns:
        episodes_data: list of dict with keys:
            - true_mass: float
            - estimated_masses: list of float (timestep ごと)
            - timesteps: list of int
            - reward: float
    """
    episodes_data = []
    
    print(f"\n{'='*70}")
    print(f"推定精度データを収集中... ({num_episodes} episodes)")
    print(f"{'='*70}\n")
    
    for ep in tqdm(range(num_episodes), desc="Collecting data"):
        obs = env.reset()
        
        # Ground truth（真の質量）を取得
        true_mass_normalized = env.current_c_phys.cpu().numpy()[0]
        true_mass = denormalize_mass(true_mass_normalized, cfg)
        
        done = False
        ep_reward = 0
        t = 0
        
        estimated_masses = []
        timesteps = []
        
        # エピソード実行
        while not done and t < max_steps:
            # アクション選択（内部で推定が行われる）
            action = agent.act(obs, t0=(t==0), eval_mode=True)
            
            # 現在の推定値を取得
            c_phys_pred = agent.estimate_physics_online()
            estimated_mass_normalized = c_phys_pred.cpu().numpy()[0, 0]
            estimated_mass = denormalize_mass(estimated_mass_normalized, cfg)
            
            estimated_masses.append(estimated_mass)
            timesteps.append(t)
            
            # 環境を進める
            obs_next, reward, done, info = env.step(action)
            agent.update_history(obs, action)
            obs = obs_next
            
            ep_reward += reward
            t += 1
        
        episodes_data.append({
            'true_mass': true_mass,
            'estimated_masses': estimated_masses,
            'timesteps': timesteps,
            'reward': ep_reward,
        })
    
    return episodes_data


def plot_estimation_scatter(episodes_data, save_path='estimation_scatter.png'):
    """推定値 vs 真値の散布図"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    true_masses = []
    estimated_masses_final = []
    colors = []
    
    for ep_data in episodes_data:
        true_mass = ep_data['true_mass']
        # 最後の50ステップの平均（安定した推定）
        estimated_mass = np.mean(ep_data['estimated_masses'][-50:])
        
        true_masses.append(true_mass)
        estimated_masses_final.append(estimated_mass)
        colors.append(ep_data['reward'])
    
    # 散布図
    scatter = ax.scatter(true_masses, estimated_masses_final, 
                        c=colors, cmap='viridis', 
                        s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # 対角線（完璧な推定）
    min_mass = min(min(true_masses), min(estimated_masses_final))
    max_mass = max(max(true_masses), max(estimated_masses_final))
    ax.plot([min_mass, max_mass], [min_mass, max_mass], 
            'r--', linewidth=3, label='Perfect Estimation', alpha=0.8)
    
    # カラーバー
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Episode Reward', fontsize=12, fontweight='bold')
    
    # 統計情報
    mae = np.mean(np.abs(np.array(true_masses) - np.array(estimated_masses_final)))
    rmse = np.sqrt(np.mean((np.array(true_masses) - np.array(estimated_masses_final))**2))
    
    ax.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('True Mass (kg)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Estimated Mass (kg)', fontsize=14, fontweight='bold')
    ax.set_title('GRU Physics Estimator: Predicted vs Ground Truth', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_estimation_convergence(episodes_data, save_path='estimation_convergence.png', num_episodes_to_plot=5):
    """エピソード内での推定値の収束"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # ランダムにいくつかのエピソードを選択
    selected_episodes = np.random.choice(len(episodes_data), 
                                        min(num_episodes_to_plot, len(episodes_data)), 
                                        replace=False)
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_episodes_to_plot))
    
    for i, ep_idx in enumerate(selected_episodes):
        ep_data = episodes_data[ep_idx]
        true_mass = ep_data['true_mass']
        timesteps = ep_data['timesteps']
        estimated_masses = ep_data['estimated_masses']
        
        # プロット
        ax.plot(timesteps, estimated_masses, 
               linewidth=2.5, alpha=0.8, color=colors[i],
               label=f'Episode {ep_idx+1}')
        
        # 真の質量を点線で表示
        ax.axhline(y=true_mass, color=colors[i], linestyle='--', 
                  linewidth=2, alpha=0.5)
    
    # Context lengthを縦線で表示
    context_length = 50
    ax.axvline(x=context_length, color='red', linestyle=':', 
              linewidth=3, alpha=0.7, label=f'Context Length ({context_length})')
    
    ax.set_xlabel('Timestep', fontsize=14, fontweight='bold')
    ax.set_ylabel('Estimated Mass (kg)', fontsize=14, fontweight='bold')
    ax.set_title('Estimation Convergence Within Episodes', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_estimation_error_distribution(episodes_data, save_path='estimation_error_dist.png'):
    """推定誤差の分布"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # データ準備
    errors_early = []  # 序盤（0-50ステップ）
    errors_mid = []    # 中盤（50-100ステップ）
    errors_late = []   # 終盤（100+ステップ）
    errors_all = []    # 全体
    
    for ep_data in episodes_data:
        true_mass = ep_data['true_mass']
        estimated_masses = np.array(ep_data['estimated_masses'])
        
        # 各期間のエラー
        if len(estimated_masses) > 0:
            errors_early.extend(estimated_masses[:50] - true_mass)
        if len(estimated_masses) > 50:
            errors_mid.extend(estimated_masses[50:100] - true_mass)
        if len(estimated_masses) > 100:
            errors_late.extend(estimated_masses[100:] - true_mass)
        errors_all.extend(estimated_masses - true_mass)
    
    # 1. 序盤のエラー分布
    axes[0, 0].hist(errors_early, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Early Stage (0-50 steps)', fontweight='bold')
    axes[0, 0].set_xlabel('Estimation Error (kg)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].text(0.05, 0.95, f'MAE: {np.mean(np.abs(errors_early)):.4f}', 
                   transform=axes[0, 0].transAxes, fontsize=11, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. 中盤のエラー分布
    axes[0, 1].hist(errors_mid, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Mid Stage (50-100 steps)', fontweight='bold')
    axes[0, 1].set_xlabel('Estimation Error (kg)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].text(0.05, 0.95, f'MAE: {np.mean(np.abs(errors_mid)):.4f}', 
                   transform=axes[0, 1].transAxes, fontsize=11, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. 終盤のエラー分布
    axes[1, 0].hist(errors_late, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Late Stage (100+ steps)', fontweight='bold')
    axes[1, 0].set_xlabel('Estimation Error (kg)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].text(0.05, 0.95, f'MAE: {np.mean(np.abs(errors_late)):.4f}', 
                   transform=axes[1, 0].transAxes, fontsize=11, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. 全体のエラー分布
    axes[1, 1].hist(errors_all, bins=50, alpha=0.7, color='mediumpurple', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('All Timesteps', fontweight='bold')
    axes[1, 1].set_xlabel('Estimation Error (kg)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].text(0.05, 0.95, f'MAE: {np.mean(np.abs(errors_all)):.4f}', 
                   transform=axes[1, 1].transAxes, fontsize=11, fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle('Estimation Error Distribution by Episode Stage', 
                 fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_estimation_by_mass_range(episodes_data, save_path='estimation_by_mass.png'):
    """質量範囲別の推定精度"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 質量範囲でビン分割
    mass_bins = [0.5, 0.8, 1.0, 1.3, 1.6, 2.0]
    bin_labels = ['0.5-0.8', '0.8-1.0', '1.0-1.3', '1.3-1.6', '1.6-2.0']
    
    # データ準備
    binned_errors = {label: [] for label in bin_labels}
    binned_final_errors = {label: [] for label in bin_labels}
    
    for ep_data in episodes_data:
        true_mass = ep_data['true_mass']
        estimated_masses = np.array(ep_data['estimated_masses'])
        
        # どのビンに属するか
        for i in range(len(mass_bins) - 1):
            if mass_bins[i] <= true_mass < mass_bins[i+1]:
                bin_label = bin_labels[i]
                # 全体のエラー
                binned_errors[bin_label].extend(np.abs(estimated_masses - true_mass))
                # 最終50ステップの平均エラー
                if len(estimated_masses) >= 50:
                    final_error = np.abs(np.mean(estimated_masses[-50:]) - true_mass)
                    binned_final_errors[bin_label].append(final_error)
                break
    
    # 1. 箱ひげ図（全体のエラー）
    ax = axes[0]
    data_to_plot = [binned_errors[label] for label in bin_labels if len(binned_errors[label]) > 0]
    labels_to_plot = [label for label in bin_labels if len(binned_errors[label]) > 0]
    
    bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True,
                    medianprops=dict(color='red', linewidth=2),
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    ax.set_xlabel('True Mass Range (kg)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Absolute Error (kg)', fontsize=13, fontweight='bold')
    ax.set_title('Estimation Error by Mass Range (All Timesteps)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. バープロット（最終エラー）
    ax = axes[1]
    mean_final_errors = [np.mean(binned_final_errors[label]) if len(binned_final_errors[label]) > 0 else 0 
                        for label in bin_labels]
    std_final_errors = [np.std(binned_final_errors[label]) if len(binned_final_errors[label]) > 0 else 0 
                       for label in bin_labels]
    
    x_pos = np.arange(len(bin_labels))
    bars = ax.bar(x_pos, mean_final_errors, yerr=std_final_errors,
                  alpha=0.7, color='coral', edgecolor='black', linewidth=2,
                  capsize=7, error_kw={'linewidth': 2})
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel('True Mass Range (kg)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (kg)', fontsize=13, fontweight='bold')
    ax.set_title('Final Estimation Error by Mass Range (Last 50 steps)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 値をバーの上に表示
    for i, (bar, val) in enumerate(zip(bars, mean_final_errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_estimation_timeline(episodes_data, save_path='estimation_timeline.png'):
    """時系列での推定誤差の平均"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 全エピソードの推定誤差を時系列で平均
    max_timesteps = max(len(ep['estimated_masses']) for ep in episodes_data)
    
    # 各タイムステップでの誤差を集計
    errors_by_timestep = [[] for _ in range(max_timesteps)]
    
    for ep_data in episodes_data:
        true_mass = ep_data['true_mass']
        estimated_masses = ep_data['estimated_masses']
        
        for t, est_mass in enumerate(estimated_masses):
            errors_by_timestep[t].append(abs(est_mass - true_mass))
    
    # 平均と標準偏差を計算
    timesteps = []
    mean_errors = []
    std_errors = []
    
    for t, errors in enumerate(errors_by_timestep):
        if len(errors) > 0:
            timesteps.append(t)
            mean_errors.append(np.mean(errors))
            std_errors.append(np.std(errors))
    
    timesteps = np.array(timesteps)
    mean_errors = np.array(mean_errors)
    std_errors = np.array(std_errors)
    
    # 1. 絶対誤差の時系列
    ax = axes[0]
    ax.plot(timesteps, mean_errors, linewidth=3, color='blue', label='Mean Absolute Error')
    ax.fill_between(timesteps, mean_errors - std_errors, mean_errors + std_errors, 
                    alpha=0.3, color='blue', label='±1 Std Dev')
    ax.axvline(x=50, color='red', linestyle=':', linewidth=3, alpha=0.7, 
              label='Context Length (50)')
    ax.set_xlabel('Timestep', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error (kg)', fontsize=13, fontweight='bold')
    ax.set_title('Estimation Error Over Time (Averaged Across All Episodes)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. 相対誤差の時系列（スムージング）
    ax = axes[1]
    # 移動平均でスムージング
    window_size = 20
    smoothed_errors = np.convolve(mean_errors, np.ones(window_size)/window_size, mode='valid')
    smoothed_timesteps = timesteps[window_size-1:]
    
    ax.plot(smoothed_timesteps, smoothed_errors, linewidth=3, color='green', 
           label=f'Smoothed MAE (window={window_size})')
    ax.axvline(x=50, color='red', linestyle=':', linewidth=3, alpha=0.7, 
              label='Context Length (50)')
    ax.set_xlabel('Timestep', fontsize=13, fontweight='bold')
    ax.set_ylabel('Smoothed Mean Absolute Error (kg)', fontsize=13, fontweight='bold')
    ax.set_title('Smoothed Estimation Error Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_summary_dashboard(episodes_data, save_path='estimation_summary.png'):
    """総合ダッシュボード"""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # データ準備
    true_masses = [ep['true_mass'] for ep in episodes_data]
    final_estimates = [np.mean(ep['estimated_masses'][-50:]) for ep in episodes_data]
    rewards = [ep['reward'] for ep in episodes_data]
    
    mae = np.mean(np.abs(np.array(true_masses) - np.array(final_estimates)))
    rmse = np.sqrt(np.mean((np.array(true_masses) - np.array(final_estimates))**2))
    
    # 1. 散布図（大）
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    scatter = ax1.scatter(true_masses, final_estimates, c=rewards, cmap='viridis',
                         s=200, alpha=0.7, edgecolors='black', linewidth=2)
    min_mass = min(min(true_masses), min(final_estimates))
    max_mass = max(max(true_masses), max(final_estimates))
    ax1.plot([min_mass, max_mass], [min_mass, max_mass], 'r--', linewidth=3, alpha=0.8)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Episode Reward', fontsize=11, fontweight='bold')
    ax1.set_xlabel('True Mass (kg)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Estimated Mass (kg)', fontsize=12, fontweight='bold')
    ax1.set_title('Estimation Accuracy', fontsize=13, fontweight='bold')
    ax1.text(0.05, 0.95, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}', 
            transform=ax1.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # 2. エラー分布
    ax2 = fig.add_subplot(gs[0, 2])
    errors = np.array(final_estimates) - np.array(true_masses)
    ax2.hist(errors, bins=20, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Error (kg)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    
    # 3. 時系列収束（サンプル）
    ax3 = fig.add_subplot(gs[1, 2])
    sample_ep = episodes_data[0]
    ax3.plot(sample_ep['timesteps'], sample_ep['estimated_masses'], 
            linewidth=2, color='blue')
    ax3.axhline(y=sample_ep['true_mass'], color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=50, color='orange', linestyle=':', linewidth=2)
    ax3.set_xlabel('Timestep', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Estimated Mass', fontsize=11, fontweight='bold')
    ax3.set_title('Sample Convergence', fontsize=12, fontweight='bold')
    
    # 4. 質量 vs 報酬
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(true_masses, rewards, s=100, alpha=0.7, color='green', edgecolors='black')
    ax4.set_xlabel('True Mass (kg)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
    ax4.set_title('Mass vs Reward', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. 誤差 vs 報酬
    ax5 = fig.add_subplot(gs[2, 1])
    abs_errors = np.abs(errors)
    ax5.scatter(abs_errors, rewards, s=100, alpha=0.7, color='purple', edgecolors='black')
    ax5.set_xlabel('Absolute Error (kg)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Episode Reward', fontsize=11, fontweight='bold')
    ax5.set_title('Error vs Reward', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. 統計サマリー
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    stats_text = f"""
    📊 Estimation Statistics
    
    Episodes: {len(episodes_data)}
    
    MAE:  {mae:.4f} kg
    RMSE: {rmse:.4f} kg
    Max Error: {np.max(abs_errors):.4f} kg
    Min Error: {np.min(abs_errors):.4f} kg
    
    Correlation (mass, reward):
    {np.corrcoef(true_masses, rewards)[0,1]:.3f}
    
    Mean Reward: {np.mean(rewards):.2f}
    Std Reward:  {np.std(rewards):.2f}
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    fig.suptitle('Model C: Physics Estimation Performance Dashboard', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def print_statistics(episodes_data):
    """統計情報を表示"""
    print("\n" + "="*70)
    print("推定精度の統計情報")
    print("="*70)
    
    true_masses = [ep['true_mass'] for ep in episodes_data]
    final_estimates = [np.mean(ep['estimated_masses'][-50:]) for ep in episodes_data]
    errors = np.array(final_estimates) - np.array(true_masses)
    abs_errors = np.abs(errors)
    
    print(f"\nエピソード数: {len(episodes_data)}")
    print(f"\n最終推定精度（最後50ステップの平均）:")
    print(f"  MAE (Mean Absolute Error):  {np.mean(abs_errors):.4f} kg")
    print(f"  RMSE (Root Mean Squared Error): {np.sqrt(np.mean(errors**2)):.4f} kg")
    print(f"  最大誤差: {np.max(abs_errors):.4f} kg")
    print(f"  最小誤差: {np.min(abs_errors):.4f} kg")
    print(f"  中央値誤差: {np.median(abs_errors):.4f} kg")
    
    # 質量範囲別
    print(f"\n質量範囲別の精度:")
    mass_ranges = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    for low, high in mass_ranges:
        mask = (np.array(true_masses) >= low) & (np.array(true_masses) < high)
        if np.sum(mask) > 0:
            range_mae = np.mean(abs_errors[mask])
            print(f"  {low:.1f}-{high:.1f} kg: MAE = {range_mae:.4f} kg (n={np.sum(mask)})")
    
    # 報酬との相関
    rewards = [ep['reward'] for ep in episodes_data]
    corr = np.corrcoef(abs_errors, rewards)[0, 1]
    print(f"\n推定誤差と報酬の相関: {corr:.3f}")
    print(f"  （負の相関 = 推定が正確なほど高報酬）")


def main():
    """メイン関数"""
    args = parse_args()
    
    print("="*70)
    print("Model C: 物理パラメータ推定精度の分析")
    print("="*70)
    
    try:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    except FileNotFoundError as e:
        print(f"\nエラー: {e}")
        return
    
    # 設定とエージェントをロード
    print("\n設定とエージェントをロード中...")
    cfg, env, agent = load_config_and_agent(checkpoint_path)
    print(f"✓ ロード完了: {checkpoint_path}")
    
    # 推定データを収集
    episodes_data = collect_estimation_data(
        agent, env, cfg, num_episodes=args.episodes, max_steps=args.max_steps
    )
    
    # 可視化
    print(f"\n{'='*70}")
    print("可視化を生成中...")
    print(f"{'='*70}\n")
    
    plot_estimation_scatter(episodes_data)
    plot_estimation_convergence(episodes_data)
    plot_estimation_error_distribution(episodes_data)
    plot_estimation_by_mass_range(episodes_data)
    plot_estimation_timeline(episodes_data)
    plot_summary_dashboard(episodes_data)
    
    # 統計情報
    print_statistics(episodes_data)
    
    print("\n" + "="*70)
    print("✓ 分析完了")
    print("="*70)
    print("\n生成されたファイル:")
    print("  1. estimation_scatter.png - 推定値 vs 真値")
    print("  2. estimation_convergence.png - エピソード内での収束")
    print("  3. estimation_error_dist.png - 誤差分布")
    print("  4. estimation_by_mass.png - 質量帯別の精度")
    print("  5. estimation_timeline.png - 時系列での誤差推移")
    print("  6. estimation_summary.png - 総合ダッシュボード")
if __name__ == '__main__':
    main()

