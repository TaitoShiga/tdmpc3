#!/usr/bin/env python
"""Utility for running the TD-MPC2 baseline training (Phase 1 / environment A).

This script automates:
  * sequential training across multiple random seeds
  * collection of evaluation curves emitted during training
  * export of the final checkpoint for downstream zero-shot evaluation

The implementation intentionally shells out to ``python -m tdmpc2.train`` so that
Hydra's configuration logic remains untouched. See ``docs/tdmpc2_baseline_plan.md``
for the accompanying experiment design.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "tdmpc2" / "baseline"
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train TD-MPC2 (MLP) on Pendulum-Swingup to establish the baseline.")
	parser.add_argument(
		"--seeds",
		type=int,
		nargs="+",
		default=[0, 1, 2],
		help="Random seeds to iterate over (default: 0 1 2).")
	parser.add_argument(
		"--steps",
		type=int,
		default=500_000,
		help="Number of environment steps for training (default: 500000).")
	parser.add_argument(
		"--task",
		type=str,
		default="pendulum-swingup",
		help="Training environment identifier (default: pendulum-swingup).")
	parser.add_argument(
		"--model-size",
		type=int,
		default=5,
		choices=[1, 5, 19, 48, 317],
		help="TD-MPC2 model size preset (default: 5).")
	parser.add_argument(
		"--exp-name",
		type=str,
		default="baseline",
		help="Experiment name recorded in log directories (default: baseline).")
	parser.add_argument(
		"--results-dir",
		type=Path,
		default=DEFAULT_RESULTS_DIR,
		help=f"Directory to collect CSV logs (default: {DEFAULT_RESULTS_DIR}).")
	parser.add_argument(
		"--checkpoints-dir",
		type=Path,
		default=DEFAULT_CHECKPOINTS_DIR,
		help=f"Directory to store checkpoints (default: {DEFAULT_CHECKPOINTS_DIR}).")
	parser.add_argument(
		"--python",
		type=str,
		default=sys.executable,
		help="Python executable to invoke (default: current interpreter).")
	parser.add_argument(
		"--extra-hydra-args",
		type=str,
		nargs="*",
		default=None,
		help="Additional key=value overrides forwarded to tdmpc2.train.")
	parser.add_argument(
		"--save-video",
		action="store_true",
		help="Enable TD-MPC2 evaluation video export during training runs.")
	parser.add_argument(
		"--skip-existing",
		action="store_true",
		help="Skip seeds whose checkpoint already exists in the checkpoints dir.")
	return parser.parse_args(argv)


def run_training(seed: int, args: argparse.Namespace) -> dict:
	log_dir = REPO_ROOT / "logs" / args.task / str(seed) / args.exp_name
	checkpoint_path = args.checkpoints_dir / f"pendulum_mass1_seed{seed}.pt"

	if args.skip_existing and checkpoint_path.exists():
		print(f"[seed={seed}] Checkpoint already exists at {checkpoint_path}, skipping.")
		return {
			"seed": seed,
			"skipped": True,
			"checkpoint": str(checkpoint_path),
			"log_dir": str(log_dir),
		}

	overrides: List[str] = [
		f"task={args.task}",
		f"seed={seed}",
		f"steps={args.steps}",
		f"model_size={args.model_size}",
		f"exp_name={args.exp_name}",
		"enable_wandb=false",
		f"save_video={'true' if args.save_video else 'false'}",
	]
	if args.extra_hydra_args:
		overrides.extend(args.extra_hydra_args)

	print(f"[seed={seed}] Starting training run...")
	cmd = [args.python, "-m", "tdmpc2.train", *overrides]
	env = os.environ.copy()
	env.setdefault("MUJOCO_GL", "egl")
	env.setdefault("TD_MPC2_ORIGINAL_CWD", str(REPO_ROOT))
	pythonpath_parts = [str(REPO_ROOT), str(REPO_ROOT / "tdmpc2")]
	if env.get("PYTHONPATH"):
		pythonpath_parts.append(env["PYTHONPATH"])
	env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
	start = datetime.now()
	subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
	duration = datetime.now() - start
	print(f"[seed={seed}] Training complete in {duration}.")

	if not log_dir.exists():
		raise FileNotFoundError(
			f"Expected log directory {log_dir} not found. "
			"Verify that hydra.run.dir is configured correctly.")

	args.results_dir.mkdir(parents=True, exist_ok=True)
	(args.results_dir / "train").mkdir(parents=True, exist_ok=True)
	args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

	# Copy eval curve CSV (if emitted)
	eval_csv = log_dir / "eval.csv"
	train_csv_target = args.results_dir / "train" / f"seed_{seed:02d}.csv"
	if eval_csv.exists():
		shutil.copyfile(eval_csv, train_csv_target)
		print(f"[seed={seed}] Stored eval curve at {train_csv_target}.")
	else:
		print(f"[seed={seed}] Warning: eval.csv not found in {log_dir}.")

	# Copy final checkpoint
	checkpoint_src = log_dir / "models" / "final.pt"
	if not checkpoint_src.exists():
		raise FileNotFoundError(
			f"Expected checkpoint at {checkpoint_src} after training.")
	shutil.copyfile(checkpoint_src, checkpoint_path)
	print(f"[seed={seed}] Stored checkpoint at {checkpoint_path}.")

	run_info = {
		"seed": seed,
		"skipped": False,
		"task": args.task,
		"steps": args.steps,
		"log_dir": str(log_dir),
		"eval_curve": str(train_csv_target) if eval_csv.exists() else None,
		"checkpoint": str(checkpoint_path),
		"started_at": start.isoformat(),
		"duration_seconds": int(duration.total_seconds()),
	}
	return run_info


def main(argv: Iterable[str] | None = None) -> None:
	args = parse_args(argv)
	args.results_dir.mkdir(parents=True, exist_ok=True)
	args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

	metadata_path = args.results_dir / "train_metadata.json"
	metadata = []
	if metadata_path.exists():
		with metadata_path.open("r", encoding="utf-8") as f:
			try:
				metadata = json.load(f)
			except json.JSONDecodeError:
				print(f"Warning: could not parse {metadata_path}, starting fresh.")

	for seed in args.seeds:
		run_info = run_training(seed, args)
		metadata = [entry for entry in metadata if entry.get("seed") != seed]
		metadata.append(run_info)
		with metadata_path.open("w", encoding="utf-8") as f:
			json.dump(sorted(metadata, key=lambda x: x["seed"]), f, indent=2)

	print("Training runs complete.")


if __name__ == "__main__":
	main()
