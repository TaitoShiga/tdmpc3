#!/usr/bin/env python
"""
Evaluate walker-walk_actuator_dynamic for 4 models and plot in one figure.

Example (explicit checkpoints):
  python scripts/evaluate_walker_actuator_dynamic_compare.py \
    --baseline checkpoints/walker/baseline.pt \
    --dr checkpoints/walker/dr.pt \
    --model-c checkpoints/walker/model_c.pt \
    --oracle checkpoints/walker/oracle.pt \
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


MODEL_ORDER = ["baseline", "dr", "c", "o"]
MODEL_LABELS = {
    "baseline": "Baseline",
    "dr": "DR",
    "c": "Model C",
    "o": "Oracle",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare models on dynamic actuator task.")
    parser.add_argument("--baseline", type=Path, help="Baseline checkpoint path.")
    parser.add_argument("--dr", type=Path, help="DR checkpoint path.")
    parser.add_argument("--model-c", type=Path, help="Model C checkpoint path.")
    parser.add_argument("--oracle", type=Path, help="Oracle checkpoint path.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing baseline.pt/dr.pt/model_c.pt/oracle.pt.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of evaluation episodes per model.",
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
        "--smooth-window",
        type=int,
        default=10,
        help="Moving average window for reward smoothing.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "results" / "walker_actuator_dynamic_compare.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=REPO_ROOT / "figures" / "walker_actuator_dynamic_compare.png",
        help="Plot output path.",
    )
    return parser.parse_args()


def resolve_from_dir(checkpoint_dir: Path):
    mapping = {}
    if checkpoint_dir is None:
        return mapping
    names = {
        "baseline": ["baseline.pt"],
        "dr": ["dr.pt"],
        "model_c": ["model_c.pt", "modelc.pt"],
        "oracle": ["oracle.pt"],
    }
    for key, candidates in names.items():
        for name in candidates:
            path = checkpoint_dir / name
            if path.exists():
                mapping[key] = path
                break
    return mapping


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
    cfg.exp_name = "eval_dynamic_compare"
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


def evaluate_model(name, checkpoint, episodes, max_steps, cfg, use_oracle, use_model_c):
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
                    "model": name,
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


def plot_compare(df, output_path, smooth_window):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Actuator scale (use first model as reference)
    df_ref = df[df["model"] == MODEL_ORDER[0]].copy()
    if not df_ref.empty:
        scale_mean = df_ref.groupby("step")["actuator_scale"].mean()
        axes[0].plot(scale_mean.index, scale_mean.values, color="C1", linewidth=2)
    axes[0].set_ylabel("Actuator scale (x)")
    axes[0].set_title("Actuator scale over time")

    # Reward lines for each model
    for idx, model in enumerate(MODEL_ORDER):
        df_model = df[df["model"] == model].copy()
        if df_model.empty:
            continue
        grouped = df_model.groupby("step")["reward"]
        mean = grouped.mean()
        std = grouped.std().fillna(0.0)
        mean_smooth = mean.rolling(window=smooth_window, min_periods=1).mean()

        axes[1].plot(mean_smooth.index, mean_smooth.values, label=MODEL_LABELS[model], linewidth=2)
        axes[1].fill_between(mean.index, mean - std, mean + std, alpha=0.15)

    axes[1].set_ylabel("Reward")
    axes[1].set_xlabel("Step")
    axes[1].legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    from_dir = resolve_from_dir(args.checkpoint_dir)

    checkpoints = {
        "baseline": args.baseline or from_dir.get("baseline"),
        "dr": args.dr or from_dir.get("dr"),
        "c": args.model_c or from_dir.get("model_c"),
        "o": args.oracle or from_dir.get("oracle"),
    }

    missing = [k for k, v in checkpoints.items() if v is None]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing checkpoint(s): {missing_str}. "
            "Provide --baseline/--dr/--model-c/--oracle or --checkpoint-dir with baseline.pt/dr.pt/model_c.pt/oracle.pt."
        )

    for name, path in checkpoints.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found for {name}: {path}")

    all_dfs = []
    for model in MODEL_ORDER:
        use_oracle = model == "o"
        use_model_c = model == "c"
        cfg = build_cfg(args.seed, args.model_size, args.max_steps, use_oracle, use_model_c)
        df_model = evaluate_model(
            name=model,
            checkpoint=checkpoints[model],
            episodes=args.episodes,
            max_steps=args.max_steps,
            cfg=cfg,
            use_oracle=use_oracle,
            use_model_c=use_model_c,
        )
        all_dfs.append(df_model)

    df = pd.concat(all_dfs, ignore_index=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    plot_compare(df, args.output_plot, args.smooth_window)

    print(f"Saved CSV: {args.output_csv}")
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
