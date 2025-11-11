#!/usr/bin/env python
"""Aggregate TD-MPC2 baseline experiment outputs (Plan/Do to Check phase).

This utility combines:
  * Phase 1 training logs (environment A) emitted by ``run_baseline_training.py``
  * Phase 2 zero-shot evaluation logs (environment B) from ``run_baseline_evaluation.py``

It produces:
  * ``results/tdmpc2/baseline/summary.csv`` with per-seed metrics and deltas
  * ``logs/tdmpc2_baseline_metadata.json`` containing reproducibility metadata
  * Matplotlib figures summarizing learning curves and zero-shot returns
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "tdmpc2" / "baseline"
DEFAULT_ASSETS_DIR = REPO_ROOT / "assets" / "plots" / "baseline"
METADATA_PATH = REPO_ROOT / "logs" / "tdmpc2_baseline_metadata.json"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Aggregate TD-MPC2 baseline training/evaluation logs.")
	parser.add_argument(
		"--results-dir",
		type=Path,
		default=DEFAULT_RESULTS_DIR,
		help=f"Directory containing baseline outputs (default: {DEFAULT_RESULTS_DIR}).")
	parser.add_argument(
		"--assets-dir",
		type=Path,
		default=DEFAULT_ASSETS_DIR,
		help=f"Directory to store generated figures (default: {DEFAULT_ASSETS_DIR}).")
	parser.add_argument(
		"--summary-path",
		type=Path,
		default=DEFAULT_RESULTS_DIR / "summary.csv",
		help="Path for the aggregated CSV summary.")
	return parser.parse_args(argv)


def load_json(path: Path) -> Optional[dict]:
	if not path.exists():
		return None
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def load_train_curve(curve_path: str) -> Optional[np.ndarray]:
	if not curve_path:
		return None
	path = Path(curve_path)
	if not path.exists():
		return None
	steps, rewards = [], []
	with path.open("r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			try:
				steps.append(int(float(row["step"])))
				rewards.append(float(row["episode_reward"]))
			except (KeyError, ValueError):
				continue
	if not steps:
		return None
	return np.stack([np.array(steps), np.array(rewards)], axis=1)


def summarize_train(curve: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
	if curve is None or len(curve) == 0:
		return {"final_reward": None, "final_step": None, "best_reward": None}
	final_step = float(curve[-1, 0])
	final_reward = float(curve[-1, 1])
	best_reward = float(np.max(curve[:, 1]))
	return {
		"final_reward": final_reward,
		"final_step": final_step,
		"best_reward": best_reward,
	}


def summarize_eval(run: dict) -> Dict[str, float]:
	return {
		"mean_return": run.get("mean_return"),
		"std_return": run.get("std_return"),
		"mean_success": run.get("mean_success"),
		"duration_seconds": run.get("duration_seconds"),
	}


def aggregate(results_dir: Path) -> Dict[str, Dict[str, Optional[float]]]:
	train_meta = load_json(results_dir / "train_metadata.json") or []
	eval_meta = load_json(results_dir / "eval_mass2_metadata.json") or {"runs": []}

	rows: Dict[int, Dict[str, Optional[float]]] = {}
	for entry in train_meta:
		if entry.get("skipped"):
			continue
		seed = entry["seed"]
		curve = load_train_curve(entry.get("eval_curve"))
		train_summary = summarize_train(curve)
		rows.setdefault(seed, {}).update({
			"train_final_return": train_summary["final_reward"],
			"train_final_step": train_summary["final_step"],
			"train_best_return": train_summary["best_reward"],
		})

	for run in eval_meta.get("runs", []):
		seed = run["seed"]
		eval_summary = summarize_eval(run)
		rows.setdefault(seed, {}).update({
			"mass2_mean_return": eval_summary["mean_return"],
			"mass2_std_return": eval_summary["std_return"],
			"mass2_mean_success": eval_summary["mean_success"],
		})

	for seed, data in rows.items():
		if data.get("train_final_return") is not None and data.get("mass2_mean_return") is not None:
			data["return_drop"] = data["train_final_return"] - data["mass2_mean_return"]
		else:
			data["return_drop"] = None
	return rows


def write_summary_csv(summary_path: Path, rows: Dict[int, Dict[str, Optional[float]]]) -> None:
	summary_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"seed",
		"train_final_step",
		"train_final_return",
		"train_best_return",
		"mass2_mean_return",
		"mass2_std_return",
		"mass2_mean_success",
		"return_drop",
	]
	with summary_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for seed in sorted(rows.keys()):
			row = {"seed": seed}
			row.update({k: rows[seed].get(k) for k in fieldnames if k != "seed"})
			writer.writerow(row)
		if rows:
			agg = {"seed": "mean"}
			for key in fieldnames[1:]:
				values = [rows[s][key] for s in rows if rows[s].get(key) is not None]
				agg[key] = float(np.mean(values)) if values else None
			writer.writerow(agg)


def plot_learning_curves(rows: Dict[int, Dict[str, Optional[float]]], results_dir: Path, assets_dir: Path) -> None:
	train_meta = load_json(results_dir / "train_metadata.json") or []
	curves = {}
	for entry in train_meta:
		if entry.get("skipped"):
			continue
		curve = load_train_curve(entry.get("eval_curve"))
		if curve is not None:
			curves[entry["seed"]] = curve

	if not curves:
		return
	assets_dir.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(6, 4))
	for seed, curve in sorted(curves.items()):
		plt.plot(curve[:, 0], curve[:, 1], label=f"seed {seed}")
	plt.xlabel("Environment steps")
	plt.ylabel("Eval episode reward (env A)")
	plt.title("TD-MPC2 Pendulum-Swingup Baseline")
	plt.legend()
	plt.tight_layout()
	out_path = assets_dir / "train_learning_curves.png"
	plt.savefig(out_path, dpi=200)
	plt.close()


def plot_zero_shot(rows: Dict[int, Dict[str, Optional[float]]], assets_dir: Path) -> None:
	seeds = sorted(rows.keys())
	values = [rows[s].get("mass2_mean_return") for s in seeds]
	if not values or any(v is None for v in values):
		return
	assets_dir.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(6, 4))
	plt.bar([str(s) for s in seeds], values, color="#d95f02")
	plt.xlabel("Seed")
	plt.ylabel("Mean episode return (env B mass=2.0)")
	plt.title("Zero-shot performance drop (TD-MPC2 MLP)")
	plt.tight_layout()
	out_path = assets_dir / "zero_shot_returns.png"
	plt.savefig(out_path, dpi=200)
	plt.close()


def collect_metadata(rows: Dict[int, Dict[str, Optional[float]]], summary_path: Path) -> dict:
	try:
		commit = (
			subprocess.run(
				["git", "rev-parse", "HEAD"],
				cwd=REPO_ROOT,
				check=True,
				capture_output=True,
				text=True,
			).stdout.strip()
		)
	except (subprocess.SubprocessError, FileNotFoundError):
		commit = None
	return {
		"generated_at": datetime.now().isoformat(),
		"repo_root": str(REPO_ROOT),
		"git_commit": commit,
		"summary_csv": str(summary_path),
		"seeds": sorted(rows.keys()),
		"metrics": rows,
	}


def write_metadata(metadata: dict) -> None:
	METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
	with METADATA_PATH.open("w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2)


def main(argv: Optional[Iterable[str]] = None) -> None:
	args = parse_args(argv)
	rows = aggregate(args.results_dir)
	if not rows:
		print("No training/evaluation logs found; summary not generated.")
		return
	write_summary_csv(args.summary_path, rows)
	plot_learning_curves(rows, args.results_dir, args.assets_dir)
	plot_zero_shot(rows, args.assets_dir)
	metadata = collect_metadata(rows, args.summary_path)
	write_metadata(metadata)
	print(f"Summary written to {args.summary_path}")
	print(f"Metadata stored in {METADATA_PATH}")


if __name__ == "__main__":
	main()
