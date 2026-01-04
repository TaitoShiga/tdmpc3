#!/usr/bin/env python
"""
Evaluate Walker actuator-scaling tasks for 4 models (baseline, dr, c, o).

Outputs a CSV compatible with analyze_results.py:
  model, seed, param, episode, return, length, success

Example:
  python evaluate/evaluate_walker_actuator_all_models.py \
    --seeds 0 1 2 3 4 \
    --episodes 30 \
    --actuator-scales 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 \
    --output results_walker_actuator.csv \
    --logs-dir logs \
    --model-size 5
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

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

# 訓練範囲: 0.4x ~ 1.4x
# In-Distribution: 0.4x ~ 1.4x (11点)
# Out-of-Distribution: 0.2x, 0.3x (軽すぎ), 1.5x, 1.6x, 1.7x (重すぎ)
DEFAULT_SCALES_IN_DIST = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
DEFAULT_SCALES_OOD = [0.2, 0.3, 1.5, 1.6, 1.7]
DEFAULT_SCALES = DEFAULT_SCALES_IN_DIST + DEFAULT_SCALES_OOD

MODEL_CONFIGS = {
    "baseline": {
        "train_task": "walker-walk",
        "exp_names": ["walker_baseline"],
        "use_oracle": False,
        "use_model_c": False,
        "available_seeds": [0],  # seed0のみ
    },
    "dr": {
        "train_task": "walker-walk_actuator_randomized",
        "exp_names": ["walker_actuator_dr"],
        "use_oracle": False,
        "use_model_c": False,
        "available_seeds": [3],  # seed3のみ
    },
    "c": {
        "train_task": "walker-walk_actuator_randomized",
        "exp_names": ["walker_actuator_model_c", "walker_actuator_modelc"],
        "use_oracle": False,
        "use_model_c": True,
        "available_seeds": [0],  # seed0のみ
    },
    "o": {
        "train_task": "walker-walk_actuator_randomized",
        "exp_names": ["walker_actuator_oracle"],
        "use_oracle": True,
        "use_model_c": False,
        "available_seeds": [0],  # seed0のみ
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Walker actuator models.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to evaluate (default: use available_seeds per model).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Episodes per (model, seed, scale) (default: 30).",
    )
    parser.add_argument(
        "--actuator-scales",
        type=float,
        nargs="+",
        default=DEFAULT_SCALES,
        help="Actuator scales to test (default: 0.2..1.7 including OOD).",
    )
    parser.add_argument(
        "--include-ood",
        action="store_true",
        default=True,
        help="Include OOD scales (0.2, 0.3, 1.5, 1.6, 1.7) in evaluation.",
    )
    parser.add_argument(
        "--in-dist-only",
        action="store_true",
        help="Evaluate only in-distribution scales (0.4-1.4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results_walker_actuator.csv",
        help="Output CSV path (default: results_walker_actuator.csv).",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=REPO_ROOT / "logs",
        help="Root directory containing logs (default: logs).",
    )
    parser.add_argument(
        "--model-size",
        type=int,
        default=5,
        choices=[1, 5, 19, 48, 317],
        help="Model size (default: 5).",
    )
    return parser.parse_args()


def build_cfg(task: str, seed: int, model_size: int, use_oracle: bool, use_model_c: bool) -> OmegaConf:
    cfg = OmegaConf.load(PKG_ROOT / "config.yaml")
    cfg.task = task
    cfg.seed = seed
    cfg.model_size = model_size
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.save_agent = False
    cfg.compile = False
    cfg.eval_episodes = 1
    cfg.steps = 1
    cfg.exp_name = "eval"
    cfg.data_dir = str(REPO_ROOT / "datasets")
    cfg.use_oracle = use_oracle
    cfg.use_model_c = use_model_c
    cfg.multitask = False
    cfg.obs = "state"
    cfg.episodic = False

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


def scale_to_task(scale: float) -> str:
    scaled = int(round(scale * 10))
    if not np.isclose(scale, scaled / 10.0, atol=1e-6):
        raise ValueError(f"Unsupported scale (must be 0.1 increments): {scale}")
    return f"walker-walk_actuator_{scaled:02d}x"


def find_checkpoint(logs_dir: Path, train_task: str, seed: int, exp_names: List[str]) -> Path:
    for exp_name in exp_names:
        candidate = logs_dir / train_task / str(seed) / exp_name / "models" / "final.pt"
        if candidate.exists():
            return candidate
    return None


def load_agent(checkpoint_path: Path, cfg: OmegaConf, use_oracle: bool, use_model_c: bool):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if use_model_c:
        agent = TDMPC2ModelC(cfg)
    elif use_oracle:
        agent = TDMPC2Oracle(cfg)
    else:
        agent = TDMPC2(cfg)

    agent.load(str(checkpoint_path))
    agent.eval()
    return agent


def evaluate_one_episode(agent, env, use_oracle: bool, use_model_c: bool, max_steps: int = 1000) -> dict:
    obs = env.reset()
    done = False
    episode_return = 0.0
    t = 0

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
                c_phys = env.current_c_phys
                action = agent.act(obs, c_phys, t0=(t == 0), eval_mode=True)
                obs, reward, done, info = env.step(action)
            else:
                action = agent.act(obs, t0=(t == 0), eval_mode=True)
                obs, reward, done, info = env.step(action)

            episode_return += float(reward)
            t += 1

    return {
        "return": episode_return,
        "length": t,
        "success": float(info.get("success", 0.0)),
    }


def evaluate_model(model_type: str, seed: int, scale: float, episodes: int, args: argparse.Namespace) -> List[dict]:
    config = MODEL_CONFIGS[model_type]
    checkpoint_path = find_checkpoint(args.logs_dir, config["train_task"], seed, config["exp_names"])
    if checkpoint_path is None:
        print(f"Warning: checkpoint not found for {model_type} seed={seed}")
        return []

    eval_task = scale_to_task(scale)
    use_oracle = config["use_oracle"]
    use_model_c = config["use_model_c"]

    cfg = build_cfg(
        task=eval_task,
        seed=seed,
        model_size=args.model_size,
        use_oracle=use_oracle,
        use_model_c=use_model_c,
    )
    set_seed(cfg.seed)

    env = make_env(cfg)
    if use_oracle or use_model_c:
        env = wrap_with_physics_param(env, cfg)

    try:
        agent = load_agent(checkpoint_path, cfg, use_oracle, use_model_c)
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")
        return []

    results = []
    for ep in range(episodes):
        result = evaluate_one_episode(agent, env, use_oracle, use_model_c)
        results.append({
            "model": model_type,
            "seed": seed,
            "param": scale,
            "episode": ep,
            "return": result["return"],
            "length": result["length"],
            "success": result["success"],
        })

    try:
        env.close()
    except Exception:
        pass

    return results


def main():
    args = parse_args()

    # OOD範囲の設定
    if args.in_dist_only:
        args.actuator_scales = [s for s in args.actuator_scales if s in DEFAULT_SCALES_IN_DIST]
    elif args.include_ood and args.actuator_scales == DEFAULT_SCALES:
        # デフォルト: In-Dist + OODの両方
        pass
    
    print("=" * 70)
    print("Walker actuator evaluation (baseline, dr, c, o)")
    print("=" * 70)
    print(f"Episodes per (model, seed, scale): {args.episodes}")
    print(f"Actuator scales: {args.actuator_scales}")
    print(f"  - In-Distribution (0.4-1.4): {[s for s in args.actuator_scales if 0.4 <= s <= 1.4]}")
    print(f"  - Out-of-Distribution: {[s for s in args.actuator_scales if s < 0.4 or s > 1.4]}")
    print(f"Logs dir: {args.logs_dir}")
    print(f"Output: {args.output}")
    print()
    print("Available seeds per model:")
    for model in ["baseline", "dr", "c", "o"]:
        avail_seeds = MODEL_CONFIGS[model]["available_seeds"]
        print(f"  - {model}: {avail_seeds}")
    print()

    all_results = []

    models = ["baseline", "dr", "c", "o"]
    
    # 総評価数を計算（モデルごとに利用可能なseedを使用）
    total_evals = sum(
        len(MODEL_CONFIGS[model]["available_seeds"]) * len(args.actuator_scales)
        for model in models
    )

    with tqdm(total=total_evals, desc="Evaluating") as pbar:
        for model in models:
            # モデルごとに利用可能なseedを使用
            available_seeds = MODEL_CONFIGS[model]["available_seeds"]
            seeds_to_eval = args.seeds if args.seeds is not None else available_seeds
            
            # 利用可能なseedのみを評価
            actual_seeds = [s for s in seeds_to_eval if s in available_seeds]
            
            if not actual_seeds:
                print(f"Warning: No valid seeds for {model}. Skipping.")
                pbar.update(len(args.actuator_scales))
                continue
            
            for seed in actual_seeds:
                for scale in args.actuator_scales:
                    # OOD判定
                    ood_label = " (OOD)" if (scale < 0.4 or scale > 1.4) else ""
                    desc = f"{model} seed={seed} scale={scale:.1f}{ood_label}"
                    pbar.set_description(desc)

                    try:
                        results = evaluate_model(model, seed, scale, args.episodes, args)
                        all_results.extend(results)

                        if results:
                            mean_return = np.mean([r["return"] for r in results])
                            print(f"  {desc}: mean_return={mean_return:.2f}")
                    except Exception as exc:
                        print(f"  Error in {desc}: {exc}")

                    pbar.update(1)

    if all_results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            fieldnames = ["model", "seed", "param", "episode", "return", "length", "success"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow({k: row[k] for k in fieldnames})

        print()
        print("=" * 70)
        print(f"Saved {len(all_results)} rows to {args.output}")
        print("=" * 70)
    else:
        print("Error: No results collected")
        sys.exit(1)


if __name__ == "__main__":
    main()
