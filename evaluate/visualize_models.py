#!/usr/bin/env python
"""
Render and save evaluation videos for Baseline / DR / Model C / Oracle agents.

Usage:
    python evaluate/visualize_models.py --mass 1.5
    python evaluate/visualize_models.py --mass 0.5 1.0 1.5 2.0 2.5 --seed 0
"""

from pathlib import Path
import argparse
import os
import sys
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "tdmpc2"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TD_MPC2_ORIGINAL_CWD", str(REPO_ROOT))

from envs import make_env
from envs.wrappers.physics_param import wrap_with_physics_param
from tdmpc2 import TDMPC2
from tdmpc2_model_c import TDMPC2ModelC
from tdmpc2_oracle import TDMPC2Oracle
from common.parser import parse_cfg


def _render(env, camera_id: int):
    """Render a frame; fall back if camera_id is unsupported."""
    try:
        return env.render(camera_id=camera_id)
    except TypeError:
        return env.render()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render comparison videos for four agents.")
    parser.add_argument(
        "--mass",
        type=float,
        nargs="+",
        default=[1.5],
        help="Mass values to evaluate (e.g., --mass 0.5 1.5 2.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed that identifies the checkpoint directory.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=REPO_ROOT / "logs_remote",
        help="Root directory that stores training checkpoints.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum episode length to record.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for the saved video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "evaluate" / "videos",
        help="Directory to save mp4 files.",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="DMControl camera id to use when rendering.",
    )
    return parser.parse_args()


def resolve_checkpoint(model_type: str, seed: int, logs_dir: Path) -> Path:
    base_dir = Path(logs_dir)
    if model_type == "baseline":
        path = base_dir / "pendulum-swingup" / str(seed) / "baseline" / "models" / "final.pt"
    elif model_type == "dr":
        path = base_dir / "pendulum-swingup-randomized" / str(seed) / "dr" / "models" / "final.pt"
    elif model_type == "c":
        path = base_dir / "pendulum-swingup-randomized" / str(seed) / "modelc" / "models" / "final.pt"
    elif model_type == "o":
        path = base_dir / "pendulum-swingup-randomized" / str(seed) / "oracle" / "models" / "final.pt"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def normalize_mass(mass: float, cfg) -> float:
    if cfg.phys_param_normalization == "standard":
        mean = 1.25
        std = 0.433
        return (mass - mean) / std
    if cfg.phys_param_normalization == "minmax":
        return (mass - 0.5) / 1.5
    return mass


def load_model(model_type: str, checkpoint_path: Path, cfg):
    if model_type in {"baseline", "dr"}:
        agent = TDMPC2(cfg)
    elif model_type == "c":
        agent = TDMPC2ModelC(cfg)
    elif model_type == "o":
        agent = TDMPC2Oracle(cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    agent.load(str(checkpoint_path))
    agent.eval()
    return agent


def create_env_for_model(model_type: str, cfg, mass: float):
    cfg.task = "pendulum-swingup" if model_type == "baseline" else "pendulum-swingup-randomized"
    cfg.obs = "state"

    env = make_env(cfg)
    if model_type != "baseline":
        env = wrap_with_physics_param(env, cfg)
    return env


def record_episode(
    agent,
    env,
    cfg,
    model_type: str,
    mass: float,
    max_steps: int,
    output_path: Path,
    fps: int = 15,
    camera_id: int = 0,
):
    """Record a single episode and save it as an MP4 file."""
    obs = env.reset()

    if model_type == "c":
        agent.reset_history()

    mass_normalized = None
    if model_type != "baseline":
        mass_normalized = normalize_mass(mass, cfg)
        env.current_c_phys = torch.tensor([mass_normalized], dtype=torch.float32)

    env.unwrapped.physics.named.model.body_mass[-1] = mass
    frames: List[np.ndarray] = [_render(env, camera_id)]

    done = False
    total_reward = 0.0
    t = 0

    if model_type == "o" and mass_normalized is None:
        mass_normalized = normalize_mass(mass, cfg)

    while not done and t < max_steps:
        if model_type == "c":
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
        elif model_type == "o":
            c_phys = torch.tensor([mass_normalized], dtype=torch.float32, device=agent.device)
            action = agent.act(obs, t0=(t == 0), eval_mode=True, c_phys=c_phys)
        else:
            action = agent.act(obs, t0=(t == 0), eval_mode=True)

        obs_next, reward, done, info = env.step(action)

        if model_type == "c":
            agent.update_history(obs, action)

        obs = obs_next
        total_reward += float(reward)
        t += 1

        frames.append(_render(env, camera_id))

    try:
        import imageio
    except ImportError:
        print("Warning: imageio not installed, skipping video save")
        return total_reward, t

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_arr = np.asarray(frames, dtype=np.uint8)
    imageio.mimsave(str(output_path), frames_arr, fps=fps)
    return total_reward, t


def build_cfg(seed: int, model_type: str):
    cfg = OmegaConf.load(PKG_ROOT / "config.yaml")
    cfg.episodic = False
    cfg.seed = seed
    cfg.compile = False
    cfg.multitask = False
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.save_agent = False
    cfg.eval_episodes = 1
    cfg.steps = 1
    cfg.exp_name = "video_eval"
    cfg.data_dir = str(REPO_ROOT / "datasets")

    cfg.use_model_c = model_type == "c"
    cfg.use_oracle = model_type == "o"

    for key in ("checkpoint", "data_dir", "gru_pretrained"):
        if cfg.get(key, None) == "???":
            setattr(cfg, key, None)

    return parse_cfg(cfg)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Rendering evaluation videos")
    print("=" * 70)
    print(f"Mass values: {args.mass}")
    print(f"Seed: {args.seed}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    for mass in args.mass:
        print(f"\n[Mass={mass}]")
        for model_type in ["baseline", "dr", "c", "o"]:
            env = None
            try:
                checkpoint_path = resolve_checkpoint(model_type, args.seed, args.logs_dir)
                print(f"  {model_type.upper()}: {checkpoint_path}")

                cfg = build_cfg(args.seed, model_type)
                env = create_env_for_model(model_type, cfg, mass)
                agent = load_model(model_type, checkpoint_path, cfg)

                output_path = output_dir / f"mass_{mass}_{model_type}.mp4"
                total_reward, steps = record_episode(
                    agent,
                    env,
                    cfg,
                    model_type,
                    mass,
                    args.max_steps,
                    output_path,
                    fps=args.fps,
                    camera_id=args.camera_id,
                )

                print(f"    ✓ saved: {output_path}  (reward={total_reward:.1f}, steps={steps})")
            except Exception as exc:
                print(f"    ✗ error for {model_type} at mass={mass}: {exc}")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

    print("\nDone. Videos are saved under:", output_dir)


if __name__ == "__main__":
    main()
