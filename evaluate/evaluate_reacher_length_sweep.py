#!/usr/bin/env python
"""
Sweep evaluation for Reacher-Hard link length scaling.
Outputs a CSV and a reward plot.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
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
from tdmpc2 import TDMPC2

DEFAULT_SCALES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
DEFAULT_CKPT_FILENAMES = (
    "final.pt",
    "latest.pt",
    "model.pt",
    "checkpoint.pt",
    "agent.pt",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep Reacher-Hard length scales.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint file or directory (required unless --plot-only).",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=DEFAULT_SCALES,
        help="Length scales to evaluate (default: 0.7..1.3).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Episodes per scale (default: 30).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    parser.add_argument(
        "--model-size",
        type=int,
        default=5,
        choices=[1, 5, 19, 48, 317],
        help="Model size if checkpoint does not include config (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results_reacher_length.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=REPO_ROOT / "reacher_length_sweep.png",
        help="Output plot path.",
    )
    parser.add_argument(
        "--plot-metric",
        type=str,
        choices=["mean", "median", "iqm"],
        default="mean",
        help="Metric to plot (default: mean).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only plot from an existing CSV and exit.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV path to plot (default: --output).",
    )
    parser.add_argument(
        "--task-prefix",
        type=str,
        choices=["hard", "four_hard"],
        default=None,
        help="Task prefix to evaluate (default: infer from checkpoint).",
    )
    return parser.parse_args()


def resolve_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path
    search_dirs = [path]
    model_dir = path / "models"
    if model_dir.is_dir():
        search_dirs.insert(0, model_dir)
    for directory in search_dirs:
        for name in DEFAULT_CKPT_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Could not find checkpoint file under {path}")


def load_checkpoint_cfg(path: Path) -> Optional[dict]:
    serialization = getattr(torch, "serialization", None)
    if serialization and hasattr(serialization, "add_safe_globals"):
        from pathlib import Path as PathClass
        extra = [PathClass]
        try:
            import pathlib
            posix_path = getattr(pathlib, "PosixPath", None)
            if posix_path is not None:
                extra.append(posix_path)
        except Exception:
            pass
        serialization.add_safe_globals(extra)
    load_kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        state = torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        state = torch.load(path, **load_kwargs)
    except RuntimeError as exc:
        if "PosixPath" in str(exc) and serialization and hasattr(serialization, "safe_globals"):
            import pathlib
            with serialization.safe_globals([Path, getattr(pathlib, "PosixPath", Path)]):
                state = torch.load(path, map_location="cpu", weights_only=False)
        else:
            raise
    cfg = None
    if isinstance(state, dict):
        cfg = state.get("cfg")
        if cfg is not None and not isinstance(cfg, dict):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    return cfg


def infer_task_prefix(checkpoint_cfg: Optional[dict], checkpoint_path: Path) -> str:
    task_hint = ""
    if checkpoint_cfg and checkpoint_cfg.get("task"):
        task_hint = str(checkpoint_cfg.get("task", ""))
    else:
        task_hint = str(checkpoint_path)
    task_hint = task_hint.replace("_", "-").lower()
    if "reacher-four-hard" in task_hint or "four-hard" in task_hint:
        return "four_hard"
    if "reacher-hard" in task_hint or "hard" in task_hint:
        return "hard"
    return "hard"


def scale_to_task(scale: float, prefix: str) -> str:
    scaled = int(round(scale * 10))
    if not np.isclose(scale, scaled / 10.0, atol=1e-6):
        raise ValueError(f"Scale must be in 0.1 increments: {scale}")
    return f"reacher-{prefix}_length_{scaled:02d}x"


def build_cfg(task: str, seed: int, model_size: int, checkpoint_cfg: Optional[dict]) -> OmegaConf:
    cfg = OmegaConf.load(PKG_ROOT / "config.yaml")
    cfg.task = task
    cfg.seed = seed
    if checkpoint_cfg and checkpoint_cfg.get("model_size") is not None:
        cfg.model_size = checkpoint_cfg["model_size"]
    else:
        cfg.model_size = model_size
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.save_agent = False
    cfg.compile = False
    cfg.eval_episodes = 1
    cfg.steps = 1
    cfg.exp_name = "eval"
    cfg.data_dir = str(REPO_ROOT / "datasets")
    cfg.multitask = False
    cfg.obs = "state"
    cfg.episodic = False
    if cfg.get("checkpoint", "???") == "???":
        cfg.checkpoint = None
    if cfg.get("data_dir", "???") == "???":
        cfg.data_dir = None
    if cfg.get("gru_pretrained", "???") == "???":
        cfg.gru_pretrained = None
    return parse_cfg(cfg)


def evaluate_scale(agent: TDMPC2, env, episodes: int) -> List[float]:
    returns = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        t = 0
        with torch.no_grad():
            while not done:
                action = agent.act(obs, t0=(t == 0), eval_mode=True)
                obs, reward, done, info = env.step(action)
                episode_return += float(reward)
                t += 1
        returns.append(episode_return)
    return returns


def compute_iqm(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    mid = values[(values >= q25) & (values <= q75)]
    if mid.size == 0:
        return float("nan")
    return float(np.mean(mid))


def plot_results(results: List[dict], plot_path: Path, metric: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plot.")
        return
    scales = [r["scale"] for r in results]
    if metric == "mean":
        ys = [r["mean_return"] for r in results]
        yerr = [r.get("std_return", 0.0) for r in results]
        yerr = np.array(yerr)
    elif metric == "median":
        ys = [r["median_return"] for r in results]
        yerr = [r.get("std_return", 0.0) for r in results]
        yerr = np.array(yerr)
    else:
        ys = [r["iqm_return"] for r in results]
        q25s = [r.get("q25_return") for r in results]
        q75s = [r.get("q75_return") for r in results]
        if all(q is not None for q in q25s + q75s):
            lower = np.array(ys) - np.array(q25s)
            upper = np.array(q75s) - np.array(ys)
            yerr = np.vstack([lower, upper])
        else:
            yerr = None
    plt.figure(figsize=(8, 5))
    plt.errorbar(scales, ys, yerr=yerr, fmt="o-", capsize=4, linewidth=2)
    plt.axvline(1.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Link length scale")
    ylabel = {
        "mean": "Mean episode return",
        "median": "Median episode return",
        "iqm": "IQM episode return",
    }[metric]
    plt.ylabel(ylabel)
    plt.title("Reacher-Hard length sweep")
    plt.grid(True, alpha=0.3)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_path}")


def load_results_from_csv(path: Path) -> List[dict]:
    results = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "scale": float(row.get("scale", "nan")),
                "mean_return": float(row.get("mean_return", "nan")),
                "std_return": float(row.get("std_return", "0")),
                "median_return": float(row.get("median_return", "nan")),
                "iqm_return": float(row.get("iqm_return", "nan")) if "iqm_return" in row else None,
                "q25_return": float(row.get("q25_return", "nan")) if "q25_return" in row else None,
                "q75_return": float(row.get("q75_return", "nan")) if "q75_return" in row else None,
            })
    return results


def main():
    args = parse_args()
    if args.plot_only:
        csv_path = args.csv or args.output
        if csv_path is None or not csv_path.exists():
            print(f"Error: CSV not found: {csv_path}")
            sys.exit(1)
        results = load_results_from_csv(csv_path)
        if not results:
            print("Error: no rows found in CSV")
            sys.exit(1)
        if args.plot_metric == "iqm" and any(r.get("iqm_return") is None for r in results):
            print("Error: CSV does not contain IQM metrics. Re-run evaluation to regenerate CSV.")
            sys.exit(1)
        plot_results(results, args.plot, args.plot_metric)
        return

    if args.checkpoint is None:
        print("Error: --checkpoint is required unless --plot-only is set.")
        sys.exit(1)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint_cfg = load_checkpoint_cfg(checkpoint_path)
    task_prefix = args.task_prefix or infer_task_prefix(checkpoint_cfg, checkpoint_path)

    print("=" * 70)
    print("Reacher-Hard length sweep")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Task prefix: {task_prefix}")
    print(f"Scales: {args.scales}")
    print(f"Episodes per scale: {args.episodes}")
    print(f"Output CSV: {args.output}")
    print(f"Plot: {args.plot}")
    print()

    all_results = []
    for scale in args.scales:
        task = scale_to_task(scale, task_prefix)
        cfg = build_cfg(task, args.seed, args.model_size, checkpoint_cfg)
        set_seed(cfg.seed)
        try:
            env = make_env(cfg)
        except Exception as exc:
            print(f"Warning: could not create task {task}: {exc}")
            continue
        agent = TDMPC2(cfg)
        agent.load(str(checkpoint_path))
        agent.eval()
        returns = evaluate_scale(agent, env, args.episodes)
        try:
            env.close()
        except Exception:
            pass
        returns_arr = np.array(returns, dtype=np.float32)
        q25 = float(np.percentile(returns_arr, 25))
        q75 = float(np.percentile(returns_arr, 75))
        iqm = compute_iqm(returns_arr)
        result = {
            "scale": scale,
            "mean_return": float(np.mean(returns_arr)),
            "std_return": float(np.std(returns_arr)),
            "min_return": float(np.min(returns_arr)),
            "max_return": float(np.max(returns_arr)),
            "median_return": float(np.median(returns_arr)),
            "iqm_return": float(iqm),
            "q25_return": q25,
            "q75_return": q75,
            "episodes": args.episodes,
        }
        all_results.append(result)
        print(f"  scale={scale:.1f}: mean_return={result['mean_return']:.2f}")

    if not all_results:
        print("Error: no results collected")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        fieldnames = [
            "scale",
            "mean_return",
            "std_return",
            "min_return",
            "max_return",
            "median_return",
            "iqm_return",
            "q25_return",
            "q75_return",
            "episodes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"Saved {len(all_results)} rows to {args.output}")

    plot_results(all_results, args.plot, args.plot_metric)


if __name__ == "__main__":
    main()
