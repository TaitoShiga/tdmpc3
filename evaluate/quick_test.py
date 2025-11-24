#!/usr/bin/env python
"""
クイックテストスクリプト

1つのモデル（baseline, seed=0, param=1.0）で5エピソードだけ評価して動作確認

使用方法:
    python evaluate/quick_test.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "tdmpc2"
for path in (REPO_ROOT, PKG_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TD_MPC2_ORIGINAL_CWD", str(REPO_ROOT))

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2


def main():
    print("="*70)
    print("クイックテスト: baseline モデル評価")
    print("="*70)
    
    # 設定
    checkpoint_path = REPO_ROOT / "logs_remote" / "pendulum-swingup" / "0" / "baseline" / "models" / "final.pt"
    
    if not checkpoint_path.exists():
        print(f"Error: チェックポイントが見つかりません: {checkpoint_path}")
        sys.exit(1)
    
    # 設定を構築
    cfg = OmegaConf.load(PKG_ROOT / "config.yaml")
    cfg.task = "pendulum-swingup"
    cfg.seed = 0
    cfg.model_size = 5
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.save_agent = False
    cfg.compile = False
    cfg.eval_episodes = 1
    cfg.steps = 1
    cfg.exp_name = "quick_test"
    cfg.data_dir = str(REPO_ROOT / "datasets")
    cfg.multitask = False
    cfg.obs = 'state'
    cfg.episodic = False
    
    # ??? を None に
    if cfg.get('checkpoint', '???') == '???':
        cfg.checkpoint = None
    if cfg.get('data_dir', '???') == '???':
        cfg.data_dir = None
    
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    
    print(f"チェックポイント: {checkpoint_path}")
    print(f"タスク: {cfg.task}")
    print()
    
    # 環境作成
    env = make_env(cfg)
    
    # エージェントロード
    agent = TDMPC2(cfg)
    agent.load(str(checkpoint_path))
    agent.eval()
    
    print("✓ エージェントロード完了")
    print()
    
    # 5エピソード評価
    returns = []
    
    for ep in range(5):
        obs = env.reset()
        done = False
        episode_return = 0.0
        t = 0
        
        with torch.no_grad():
            while not done and t < 1000:
                action = agent.act(obs, t0=(t == 0), eval_mode=True)
                obs, reward, done, info = env.step(action)
                episode_return += float(reward)
                t += 1
        
        returns.append(episode_return)
        print(f"Episode {ep}: return={episode_return:.2f}, length={t}")
    
    print()
    print("="*70)
    print("結果サマリー")
    print("="*70)
    print(f"平均リターン: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"最小/最大: {np.min(returns):.2f} / {np.max(returns):.2f}")
    print()
    print("✓ テスト成功！")


if __name__ == "__main__":
    main()

