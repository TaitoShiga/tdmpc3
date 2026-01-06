#!/usr/bin/env python
"""
Evaluate walker-walk_actuator_dynamic and save time-series plots.

Example:
  python scripts/evaluate_walker_actuator_dynamic.py \
    --checkpoint logs/walker-walk/0/walker_baseline/models/final.pt \
    --episodes 3 \
    --max-steps 800
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "tdmpc2"
for path in (REPO_ROOT, PKG_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TD_MPC2_ORIGINAL_CWD", str(REPO_ROOT))

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from envs.wrappers.physics_param import wrap_with_physics_param
from tdmpc2 import TDMPC2
from tdmpc2_oracle import TDMPC2Oracle
from tdmpc2_model_c import TDMPC2ModelC


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dynamic actuator task.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--model-size",
        type=int,
        default=5,
        choices=[1, 5, 19, 48, 317],
        help="Model size used for training.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=800,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--plot-episode",
        type=int,
        default=0,
        help="Episode index to plot.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Moving average window for reward smoothing.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "results" / "walker_actuator_dynamic_timeseries.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=REPO_ROOT / "figures" / "walker_actuator_dynamic_timeseries_ep0.png",
        help="Plot output path.",
    )
    parser.add_argument(
        "--use-oracle",
        action="store_true",
        help="Use Oracle model for evaluation.",
    )
    parser.add_argument(
        "--use-model-c",
        action="store_true",
        help="Use Model C for evaluation.",
    )
    return parser.parse_args()


def build_cfg(seed, model_size, max_steps, use_oracle, use_model_c):
    cfg = OmegaConf.load(PKG_ROOT / "config.yaml")
    cfg.task = "walker-walk_actuator_dynamic"
    cfg.seed = seed
    cfg.model_size = model_size
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.save_agent = False
    cfg.compile = False
    cfg.eval_episodes = 1
    cfg.steps = 1
    cfg.exp_name = "eval_dynamic"
    cfg.data_dir = str(REPO_ROOT / "datasets")
    cfg.multitask = False
    cfg.obs = "state"
    cfg.episodic = False
    cfg.max_episode_steps = max_steps
    cfg.use_oracle = use_oracle
    cfg.use_model_c = use_model_c

    if use_oracle or use_model_c:
        cfg.c_phys_dim = 1
        cfg.phys_param_type = "actuator"
        cfg.phys_param_normalization = "standard"

    if cfg.get("checkpoint", "???") == "???":
        cfg.checkpoint = None
    if cfg.get("data_dir", "???") == "???":
        cfg.data_dir = None
    if cfg.get("gru_pretrained", "???") == "???":
        cfg.gru_pretrained = None

    cfg = parse_cfg(cfg)
    return cfg


def get_actuator_scale(env):
    try:
        task = env.unwrapped.task
        return float(getattr(task, "current_actuator_scale"))
    except Exception:
        return np.nan


def load_agent(cfg, checkpoint, use_oracle, use_model_c):
    if use_model_c:
        agent = TDMPC2ModelC(cfg)
    elif use_oracle:
        agent = TDMPC2Oracle(cfg)
    else:
        agent = TDMPC2(cfg)
    agent.load(str(checkpoint))
    agent.eval()
    return agent


def evaluate(cfg, checkpoint, episodes, max_steps, use_oracle, use_model_c):
    set_seed(cfg.seed)
    env = make_env(cfg)
    if use_oracle or use_model_c:
        env = wrap_with_physics_param(env, cfg)

    agent = load_agent(cfg, checkpoint, use_oracle, use_model_c)

    records = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        t = 0
        ep_return = 0.0

        if use_model_c:
            agent.reset_history()

        with torch.no_grad():
            while not done and t < max_steps:
                if use_model_c:
                    action = agent.act(obs, t0=(t == 0), eval_mode=True)
                    obs_next, reward, done, info = env.step(action)
                    agent.update_history(obs, action)
                    obs = obs_next
                elif use_oracle:
                    action = agent.act(obs, env.current_c_phys, t0=(t == 0), eval_mode=True)
                    obs, reward, done, info = env.step(action)
                else:
                    action = agent.act(obs, t0=(t == 0), eval_mode=True)
                    obs, reward, done, info = env.step(action)

                t += 1
                reward_val = float(reward)
                ep_return += reward_val
                records.append({
                    "episode": ep,
                    "step": t,
                    "reward": reward_val,
                    "return": ep_return,
                    "actuator_scale": get_actuator_scale(env),
                })

    try:
        env.close()
    except Exception:
        pass

    return pd.DataFrame.from_records(records)


def plot_timeseries(df, episode, output_path, smooth_window):
    df_ep = df[df["episode"] == episode].copy()
    if df_ep.empty:
        raise ValueError(f"No data for episode {episode}")

    df_ep["reward_smooth"] = (
        df_ep["reward"].rolling(window=smooth_window, min_periods=1).mean()
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(df_ep["step"], df_ep["actuator_scale"], color="C1", linewidth=2)
    axes[0].set_ylabel("Actuator scale (x)")
    axes[0].set_title(f"Actuator scale over time (episode {episode})")

    axes[1].plot(df_ep["step"], df_ep["reward"], color="C0", alpha=0.3, label="Reward")
    axes[1].plot(df_ep["step"], df_ep["reward_smooth"], color="C0", linewidth=2, label="Reward (smooth)")
    axes[1].set_ylabel("Reward")
    axes[1].set_xlabel("Step")
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    if args.use_oracle and args.use_model_c:
        raise ValueError("Cannot use both Oracle and Model C modes.")

    checkpoint = args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg = build_cfg(
        seed=args.seed,
        model_size=args.model_size,
        max_steps=args.max_steps,
        use_oracle=args.use_oracle,
        use_model_c=args.use_model_c,
    )

    df = evaluate(
        cfg=cfg,
        checkpoint=checkpoint,
        episodes=args.episodes,
        max_steps=args.max_steps,
        use_oracle=args.use_oracle,
        use_model_c=args.use_model_c,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    plot_timeseries(df, args.plot_episode, args.output_plot, args.smooth_window)

    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
