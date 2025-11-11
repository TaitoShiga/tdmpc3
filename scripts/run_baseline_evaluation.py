#!/usr/bin/env python
"""Zero-shot robustness evaluation for the TD-MPC2 baseline (Phase 2 / environment B).

Given a set of checkpoints trained on the standard Pendulum-Swingup task, this script:
  * builds the modified mass environment (pendulum-swingup-mass2)
  * runs inference-only rollouts without updating model weights
  * records per-episode returns and summary statistics for downstream analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
import sys

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
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "tdmpc2" / "baseline"
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
DEFAULT_ASSETS_DIR = REPO_ROOT / "assets" / "plots" / "baseline" / "eval_mass2"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate TD-MPC2 checkpoints zero-shot on pendulum-swingup-mass2.")
	parser.add_argument(
		"--checkpoints-dir",
		type=Path,
		default=DEFAULT_CHECKPOINTS_DIR,
		help=f"Directory containing Phase 1 checkpoints (default: {DEFAULT_CHECKPOINTS_DIR}).")
	parser.add_argument(
		"--results-dir",
		type=Path,
		default=DEFAULT_RESULTS_DIR,
		help=f"Directory to store evaluation outputs (default: {DEFAULT_RESULTS_DIR}).")
	parser.add_argument(
		"--seeds",
		type=int,
		nargs="+",
		default=[0, 1, 2],
		help="Random seeds to evaluate (default: 0 1 2).")
	parser.add_argument(
		"--eval-episodes",
		type=int,
		default=100,
		help="Number of evaluation episodes per seed (default: 100).")
	parser.add_argument(
		"--task",
		type=str,
		default="pendulum-swingup-mass2",
		help="Evaluation task identifier (default: pendulum-swingup-mass2).")
	parser.add_argument(
		"--model-size",
		type=int,
		default=5,
		choices=[1, 5, 19, 48, 317],
		help="TD-MPC2 model size preset (default: 5).")
	parser.add_argument(
	"--exp-name",
	type=str,
	default="baseline_mass2_eval",
	help="Experiment name used for the evaluation run directory (default: baseline_mass2_eval).")
	parser.add_argument(
		"--plots",
		type=str,
		nargs="*",
		choices={"hist", "reward_hist", "length_hist"},
		default=None,
		help="Generate diagnostic plots (hist/length_hist). Saved under assets/plots/baseline/eval_mass2/")
	parser.add_argument(
		"--save-video",
		action="store_true",
		help="Save the first evaluation episode per seed as MP4.")
	return parser.parse_args(argv)


def build_cfg(
	task: str,
	seed: int,
	model_size: int,
	checkpoint: Path,
	exp_name: str,
	save_video: bool,
) -> OmegaConf:
	cfg = OmegaConf.load(REPO_ROOT / "tdmpc2" / "config.yaml")
	cfg.task = task
	cfg.seed = seed
	cfg.model_size = model_size
	cfg.checkpoint = str(checkpoint)
	cfg.enable_wandb = False
	cfg.save_video = bool(save_video)
	cfg.save_agent = False
	cfg.compile = False
	cfg.eval_episodes = 1  # Unused in this script but satisfies config requirements.
	cfg.steps = max(int(cfg.steps), 1)
	cfg.exp_name = exp_name
	cfg.data_dir = str(REPO_ROOT / "datasets")
	cfg.wandb_project = "none"
	cfg.wandb_entity = "none"
	return cfg


def evaluate_seed(
	seed: int,
	args: argparse.Namespace,
) -> dict:
	checkpoint = args.checkpoints_dir / f"pendulum_mass1_seed{seed}.pt"
	if not checkpoint.exists():
		raise FileNotFoundError(f"Checkpoint not found for seed {seed}: {checkpoint}")

	cfg = build_cfg(
		task=args.task,
		seed=seed,
		model_size=args.model_size,
		checkpoint=checkpoint,
		exp_name=args.exp_name,
		save_video=args.save_video,
	)
	cfg = parse_cfg(cfg)

	if not torch.cuda.is_available():
		raise RuntimeError("CUDA device required for TD-MPC2 evaluation.")
	set_seed(cfg.seed)
	env = make_env(cfg)

	agent = TDMPC2(cfg)
	agent.load(str(checkpoint))
	agent.eval()

	per_episode = []
	video_dir = args.results_dir / "eval_mass2" / "videos" if cfg.save_video else None
	if video_dir:
		video_dir.mkdir(parents=True, exist_ok=True)
	start = datetime.now()
	for episode_idx in range(args.eval_episodes):
		obs = env.reset()
		done = False
		episode_return = 0.0
		t = 0
		frames = [env.render()] if video_dir and episode_idx == 0 else None
		with torch.no_grad():
			while not done:
				action = agent.act(obs, t0=(t == 0), eval_mode=True)
				obs, reward, done, info = env.step(action)
				episode_return += float(reward)
				t += 1
				if frames is not None:
					frames.append(env.render())
		per_episode.append(
			{
				"episode": episode_idx,
				"return": episode_return,
				"length": t,
				"success": float(info.get("success", 0.0)),
			}
		)
		if frames is not None:
			try:
				import imageio
			except ImportError:
				print("Install imageio to enable video export.")
				frames = None
			else:
				fp = video_dir / f"seed_{seed:02d}_episode_{episode_idx:03d}.mp4"
				imageio.mimsave(fp, np.asarray(frames, dtype=np.uint8), fps=15)
				print(f"[seed={seed}] Saved evaluation video to {fp}")

	returns = np.array([row["return"] for row in per_episode], dtype=np.float64)
	success = np.array([row["success"] for row in per_episode], dtype=np.float64)
	lengths = np.array([row["length"] for row in per_episode], dtype=np.int32)

	duration = datetime.now() - start

	return {
		"seed": seed,
		"checkpoint": str(checkpoint),
		"mean_return": float(np.mean(returns)),
		"std_return": float(np.std(returns, ddof=0)),
		"min_return": float(np.min(returns)),
		"max_return": float(np.max(returns)),
		"mean_success": float(np.mean(success)),
		"mean_length": float(np.mean(lengths)),
		"episodes": per_episode,
		"duration_seconds": int(duration.total_seconds()),
	}


def write_episode_csv(seed: int, results: dict, args: argparse.Namespace) -> Path:
	target_dir = args.results_dir / "eval_mass2"
	target_dir.mkdir(parents=True, exist_ok=True)
	out_path = target_dir / f"seed_{seed:02d}.csv"
	with out_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["episode", "return", "length", "success"])
		writer.writeheader()
		for row in results["episodes"]:
			writer.writerow(row)
	return out_path


def update_summary(summary_path: Path, seed_results: List[dict], task: str) -> None:
	payload = {
		"task": task,
		"seeds": [res["seed"] for res in seed_results],
		"updated_at": datetime.now().isoformat(),
		"runs": seed_results,
	}
	with summary_path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def generate_plots(seed_results: List[dict], plots: Optional[List[str]]) -> None:
	if not seed_results:
		return
	if not plots:
		return
	plot_set = set(plots)
	if "hist" in plot_set:
		plot_set.update({"reward_hist", "length_hist"})
		plot_set.discard("hist")
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except ImportError:
		print("Matplotlib not installed; skipping plot generation.")
		return

	assets_dir = DEFAULT_ASSETS_DIR
	assets_dir.mkdir(parents=True, exist_ok=True)

	returns_by_seed = {
		res["seed"]: [episode["return"] for episode in res["episodes"]]
		for res in seed_results
	}
	lengths_by_seed = {
		res["seed"]: [episode["length"] for episode in res["episodes"]]
		for res in seed_results
	}

	if "reward_hist" in plot_set:
		plt.figure(figsize=(6, 4))
		for seed, values in sorted(returns_by_seed.items()):
			plt.hist(values, bins=20, alpha=0.4, label=f"seed {seed}")
		plt.xlabel("Episode return (env B)")
		plt.ylabel("Count")
		plt.title("Zero-shot return distribution")
		plt.legend()
		plt.tight_layout()
		out_path = assets_dir / "reward_hist.png"
		plt.savefig(out_path, dpi=200)
		plt.close()
		print(f"Saved reward histogram to {out_path}")

	if "length_hist" in plot_set:
		plt.figure(figsize=(6, 4))
		for seed, values in sorted(lengths_by_seed.items()):
			plt.hist(values, bins=20, alpha=0.4, label=f"seed {seed}")
		plt.xlabel("Episode length (env steps)")
		plt.ylabel("Count")
		plt.title("Zero-shot episode length distribution")
		plt.legend()
		plt.tight_layout()
		out_path = assets_dir / "length_hist.png"
		plt.savefig(out_path, dpi=200)
		plt.close()
		print(f"Saved episode-length histogram to {out_path}")


def main(argv: Optional[Iterable[str]] = None) -> None:
	args = parse_args(argv)
	args.results_dir.mkdir(parents=True, exist_ok=True)
	args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

	summary_path = args.results_dir / "eval_mass2_metadata.json"
	all_results = []
	sorted_results: List[dict] = []
	for seed in args.seeds:
		print(f"[seed={seed}] Evaluating checkpoint...")
		results = evaluate_seed(seed, args)
		write_episode_csv(seed, results, args)
		all_results = [res for res in all_results if res["seed"] != seed]
		all_results.append(results)
		sorted_results = sorted(all_results, key=lambda x: x["seed"])
		update_summary(summary_path, sorted_results, args.task)
		print(
			f"[seed={seed}] mean_return={results['mean_return']:.2f}, "
			f"std_return={results['std_return']:.2f}"
		)
	generate_plots(sorted_results, args.plots)
	print("Zero-shot evaluation complete.")


if __name__ == "__main__":
	main()
