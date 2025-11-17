"""
チェックポイントの評価スクリプト

各モデルのチェックポイントを複数エピソードで評価。

使用方法:
    python evaluate_checkpoints.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tdmpc2'))

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import hydra
from omegaconf import OmegaConf

from envs import make_env
from envs.wrappers.physics_param import wrap_with_physics_param
from tdmpc2 import TDMPC2
from tdmpc2_oracle import TDMPC2Oracle
from tdmpc2_model_c import TDMPC2ModelC
from common.parser import parse_cfg


def load_config(use_oracle=False, use_model_c=False):
    """Hydra configをロード"""
    # config.yamlをロード
    config_path = Path(__file__).parent / 'tdmpc2' / 'config.yaml'
    cfg = OmegaConf.load(config_path)
    
    # 評価用の設定を上書き
    cfg.task = 'pendulum-swingup-randomized'
    cfg.obs = 'state'
    cfg.episodic = False
    cfg.seed = 0
    cfg.use_oracle = use_oracle
    cfg.use_model_c = use_model_c
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
    
    return cfg


def create_agent(cfg, checkpoint_path):
    """エージェントを作成してチェックポイントをロード"""
    if cfg.use_model_c:
        agent = TDMPC2ModelC(cfg)
    elif cfg.use_oracle:
        agent = TDMPC2Oracle(cfg)
    else:
        agent = TDMPC2(cfg)
    
    # チェックポイントをロード
    agent.load(checkpoint_path)
    agent.eval()
    
    return agent


def evaluate_checkpoint(agent, env, num_episodes=10, use_oracle=False, use_model_c=False):
    """
    チェックポイントを評価
    
    Args:
        agent: エージェント
        env: 環境
        num_episodes: 評価エピソード数
        use_oracle: Oracleモードかどうか
        use_model_c: Model Cモードかどうか
    
    Returns:
        results: {'rewards': [...], 'successes': [...], 'lengths': [...]}
    """
    rewards = []
    successes = []
    lengths = []
    
    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset()
        done = False
        ep_reward = 0
        t = 0
        
        # Model C の場合は履歴をリセット
        if use_model_c:
            agent.reset_history()
        
        while not done:
            # アクション選択
            if use_model_c:
                action = agent.act(obs, t0=(t==0), eval_mode=True)
                agent.update_history(obs, action)
            elif use_oracle:
                c_phys = env.current_c_phys
                action = agent.act(obs, c_phys, t0=(t==0), eval_mode=True)
            else:
                action = agent.act(obs, t0=(t==0), eval_mode=True)
            
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            t += 1
            
            if t >= 500:  # タイムアウト
                break
        
        rewards.append(ep_reward)
        successes.append(info.get('success', 0))
        lengths.append(t)
    
    return {
        'rewards': np.array(rewards),
        'successes': np.array(successes),
        'lengths': np.array(lengths),
    }


def main():
    """メイン関数"""
    print("="*70)
    print("チェックポイント評価")
    print("="*70)
    
    # 評価する設定
    checkpoints = [
        {
            'name': 'Model B (DR)',
            'path': 'logs/pendulum-swingup-randomized/0/dr/models/final.pt',
            'use_oracle': False,
            'use_model_c': False,
        },
        {
            'name': 'Model C',
            'path': 'tdmpc2/logs/pendulum-swingup-randomized/0/default/models/final.pt',
            'use_oracle': False,
            'use_model_c': True,
        },
        {
            'name': 'Model O',
            'path': 'tdmpc2/logs/pendulum-swingup-randomized/0/oracle_100k/models/final.pt',
            'use_oracle': True,
            'use_model_c': False,
        },
    ]
    
    num_episodes = 20  # 評価エピソード数
    
    results_all = []
    
    for checkpoint_info in checkpoints:
        print(f"\n{'='*70}")
        print(f"評価中: {checkpoint_info['name']}")
        print(f"{'='*70}")
        
        checkpoint_path = Path(checkpoint_info['path'])
        
        if not checkpoint_path.exists():
            print(f"✗ チェックポイントが見つかりません: {checkpoint_path}")
            continue
        
        # 設定をロード
        cfg = load_config(
            use_oracle=checkpoint_info['use_oracle'],
            use_model_c=checkpoint_info['use_model_c']
        )
        
        # 環境を作成（これでcfgにobs_shape、action_dimなどが設定される）
        env = make_env(cfg)
        if cfg.use_oracle or cfg.use_model_c:
            env = wrap_with_physics_param(env, cfg)
        
        # エージェントを作成してロード
        try:
            agent = create_agent(cfg, checkpoint_path)
            print(f"✓ チェックポイントをロード: {checkpoint_path}")
        except Exception as e:
            print(f"✗ チェックポイントのロードに失敗: {e}")
            continue
        
        # 評価
        results = evaluate_checkpoint(
            agent, env, num_episodes,
            use_oracle=cfg.use_oracle,
            use_model_c=cfg.use_model_c
        )
        
        # 統計情報
        mean_reward = results['rewards'].mean()
        std_reward = results['rewards'].std()
        mean_success = results['successes'].mean()
        mean_length = results['lengths'].mean()
        
        print(f"\n結果:")
        print(f"  平均報酬: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"  成功率: {mean_success*100:.1f}%")
        print(f"  平均長さ: {mean_length:.1f}")
        print(f"  最小/最大報酬: {results['rewards'].min():.2f} / {results['rewards'].max():.2f}")
        
        results_all.append({
            'name': checkpoint_info['name'],
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_success': mean_success,
            'mean_length': mean_length,
            'min_reward': results['rewards'].min(),
            'max_reward': results['rewards'].max(),
        })
    
    # 結果のまとめ
    print("\n" + "="*70)
    print("評価結果まとめ")
    print("="*70)
    
    df = pd.DataFrame(results_all)
    print(df.to_string(index=False))
    
    # CSVに保存
    df.to_csv('checkpoint_evaluation.csv', index=False)
    print(f"\n✓ 結果を保存: checkpoint_evaluation.csv")
    
    # 性能比較
    if len(results_all) >= 2:
        print("\n" + "="*70)
        print("性能比較")
        print("="*70)
        
        for i in range(len(results_all)):
            for j in range(i+1, len(results_all)):
                model_i = results_all[i]
                model_j = results_all[j]
                
                diff = model_j['mean_reward'] - model_i['mean_reward']
                pct = (diff / model_i['mean_reward']) * 100
                
                print(f"{model_j['name']} vs {model_i['name']}: "
                      f"{'+' if diff > 0 else ''}{diff:.2f} ({'+' if pct > 0 else ''}{pct:.1f}%)")


if __name__ == '__main__':
    main()

