#!/usr/bin/env python
"""Transformer-based TD-MPC2 training script

Domain Randomization環境でTransformerモデルを訓練し、
In-Context Learningによる適応能力を獲得させる。
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
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "tdmpc2" / "transformer"
DEFAULT_CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train Transformer-based TD-MPC2 on Domain Randomized Pendulum-Swingup")
	parser.add_argument(
		"--seeds",
		type=int,
		nargs="+",
		default=[0, 1, 2],
		help="Random seeds to iterate over (default: 0 1 2)")
	parser.add_argument(
		"--steps",
		type=int,
		default=500_000,
		help="Number of environment steps for training (default: 500000)")
	parser.add_argument(
		"--task",
		type=str,
		default="pendulum-swingup-randomized",
		help="Training environment (default: pendulum-swingup-randomized)")
	parser.add_argument(
		"--model-size",
		type=int,
		default=5,
		choices=[1, 5, 19, 48, 317])
	parser.add_argument(
		"--context-length",
		type=int,
		default=50,
		help="Context length for Transformer (default: 50)")
	parser.add_argument(
		"--transformer-layers",
		type=int,
		default=4,
		help="Number of Transformer layers (default: 4)")
	parser.add_argument(
		"--transformer-heads",
		type=int,
		default=8,
		help="Number of attention heads (default: 8)")
	parser.add_argument(
		"--exp-name",
		type=str,
		default="transformer_dr",
		help="Experiment name (default: transformer_dr)")
	parser.add_argument(
		"--results-dir",
		type=Path,
		default=DEFAULT_RESULTS_DIR,
		help=f"Directory to collect CSV logs (default: {DEFAULT_RESULTS_DIR})")
	parser.add_argument(
		"--checkpoints-dir",
		type=Path,
		default=DEFAULT_CHECKPOINTS_DIR,
		help=f"Directory to store checkpoints (default: {DEFAULT_CHECKPOINTS_DIR})")
	parser.add_argument(
		"--python",
		type=str,
		default=sys.executable)
	parser.add_argument(
		"--extra-hydra-args",
		type=str,
		nargs="*",
		default=None)
	parser.add_argument(
		"--save-video",
		action="store_true")
	parser.add_argument(
		"--skip-existing",
		action="store_true")
	return parser.parse_args(argv)


def run_training(seed: int, args: argparse.Namespace) -> dict:
	"""Run training for a single seed"""
	log_dir = REPO_ROOT / "logs" / args.task / str(seed) / args.exp_name
	checkpoint_path = args.checkpoints_dir / f"pendulum_transformer_dr_seed{seed}.pt"
	
	if args.skip_existing and checkpoint_path.exists():
		print(f"[seed={seed}] Checkpoint already exists, skipping.")
		return {
			"seed": seed,
			"skipped": True,
			"checkpoint": str(checkpoint_path),
			"log_dir": str(log_dir),
		}
	
	# Hydra overrides
	overrides: List[str] = [
		f"task={args.task}",
		f"seed={seed}",
		f"steps={args.steps}",
		f"model_size={args.model_size}",
		f"exp_name={args.exp_name}",
		f"context_length={args.context_length}",
		f"transformer_layers={args.transformer_layers}",
		f"transformer_heads={args.transformer_heads}",
		"enable_wandb=false",
		f"save_video={'true' if args.save_video else 'false'}",
		# Transformer用の設定
		"use_transformer=true",
	]
	if args.extra_hydra_args:
		overrides.extend(args.extra_hydra_args)
	
	print(f"[seed={seed}] Starting Transformer training...")
	print(f"  Task: {args.task}")
	print(f"  Context length: {args.context_length}")
	print(f"  Transformer layers: {args.transformer_layers}")
	print(f"  Attention heads: {args.transformer_heads}")
	
	# 訓練スクリプトを実行
	# Note: tdmpc2/train.pyを拡張してTransformer対応させる必要がある
	cmd = [args.python, "-m", "tdmpc2.train_transformer", *overrides]
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
		raise FileNotFoundError(f"Expected log directory {log_dir} not found.")
	
	args.results_dir.mkdir(parents=True, exist_ok=True)
	(args.results_dir / "train").mkdir(parents=True, exist_ok=True)
	args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
	
	# Copy eval curve CSV
	eval_csv = log_dir / "eval.csv"
	train_csv_target = args.results_dir / "train" / f"seed_{seed:02d}.csv"
	if eval_csv.exists():
		shutil.copyfile(eval_csv, train_csv_target)
		print(f"[seed={seed}] Stored eval curve at {train_csv_target}.")
	else:
		print(f"[seed={seed}] Warning: eval.csv not found.")
	
	# Copy checkpoint
	checkpoint_src = log_dir / "models" / "final.pt"
	if not checkpoint_src.exists():
		raise FileNotFoundError(f"Expected checkpoint at {checkpoint_src}")
	shutil.copyfile(checkpoint_src, checkpoint_path)
	print(f"[seed={seed}] Stored checkpoint at {checkpoint_path}.")
	
	return {
		"seed": seed,
		"skipped": False,
		"task": args.task,
		"steps": args.steps,
		"context_length": args.context_length,
		"transformer_layers": args.transformer_layers,
		"transformer_heads": args.transformer_heads,
		"log_dir": str(log_dir),
		"eval_curve": str(train_csv_target) if eval_csv.exists() else None,
		"checkpoint": str(checkpoint_path),
		"started_at": start.isoformat(),
		"duration_seconds": int(duration.total_seconds()),
	}


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
	
	print("\n✅ Transformer training runs complete.")
	print(f"Results saved in: {args.results_dir}")
	print(f"Checkpoints saved in: {args.checkpoints_dir}")


if __name__ == "__main__":
	main()

