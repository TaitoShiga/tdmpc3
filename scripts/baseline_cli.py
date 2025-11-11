#!/usr/bin/env python
"""Unified CLI for TD-MPC2 baseline experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "tdmpc2"
for path in (REPO_ROOT, PKG_ROOT):
	if str(path) not in sys.path:
		sys.path.insert(0, str(path))
BASELINE_RESULTS_DIR = REPO_ROOT / "results" / "tdmpc2" / "baseline"
BASELINE_ASSETS_DIR = REPO_ROOT / "assets" / "plots" / "baseline"
BASELINE_SUMMARY_PATH = BASELINE_RESULTS_DIR / "summary.csv"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"


def _path_to_str(path: Optional[Path]) -> Optional[str]:
	return None if path is None else str(path)


def _maybe_extend(argv: List[str], flag: str, values: Optional[Iterable[str | int | Path]]) -> None:
	if values is None:
		return
	val_list = list(values)
	if not val_list:
		return
	argv.append(flag)
	argv.extend([_path_to_str(v) if isinstance(v, Path) else str(v) for v in val_list])


def build_train_argv(args: argparse.Namespace) -> List[str]:
	argv: List[str] = []
	_maybe_extend(argv, "--seeds", args.seeds)
	if args.steps is not None:
		argv.extend(["--steps", str(args.steps)])
	if args.task is not None:
		argv.extend(["--task", args.task])
	if args.model_size is not None:
		argv.extend(["--model-size", str(args.model_size)])
	if args.exp_name is not None:
		argv.extend(["--exp-name", args.exp_name])
	if args.results_dir is not None:
		argv.extend(["--results-dir", _path_to_str(args.results_dir)])
	if args.checkpoints_dir is not None:
		argv.extend(["--checkpoints-dir", _path_to_str(args.checkpoints_dir)])
	if args.python is not None and args.python != sys.executable:
		argv.extend(["--python", args.python])
	_maybe_extend(argv, "--extra-hydra-args", args.extra_hydra_args)
	if getattr(args, "save_video", False):
		argv.append("--save-video")
	if args.skip_existing:
		argv.append("--skip-existing")
	return argv


def build_eval_argv(args: argparse.Namespace) -> List[str]:
	argv: List[str] = []
	_maybe_extend(argv, "--seeds", args.seeds)
	if args.eval_episodes is not None:
		argv.extend(["--eval-episodes", str(args.eval_episodes)])
	if args.task is not None:
		argv.extend(["--task", args.task])
	if args.model_size is not None:
		argv.extend(["--model-size", str(args.model_size)])
	if args.exp_name is not None:
		argv.extend(["--exp-name", args.exp_name])
	if args.results_dir is not None:
		argv.extend(["--results-dir", _path_to_str(args.results_dir)])
	if args.checkpoints_dir is not None:
		argv.extend(["--checkpoints-dir", _path_to_str(args.checkpoints_dir)])
	if getattr(args, "save_video", False):
		argv.append("--save-video")
	_maybe_extend(argv, "--plots", getattr(args, "plots", None))
	return argv


def build_analyze_argv(args: argparse.Namespace) -> List[str]:
	argv: List[str] = []
	if args.results_dir is not None:
		argv.extend(["--results-dir", _path_to_str(args.results_dir)])
	if args.assets_dir is not None:
		argv.extend(["--assets-dir", _path_to_str(args.assets_dir)])
	if args.summary_path is not None:
		argv.extend(["--summary-path", _path_to_str(args.summary_path)])
	return argv


def run_train(argv: List[str]) -> None:
	from scripts import run_baseline_training as train_mod

	train_mod.main(argv)


def run_eval(argv: List[str]) -> None:
	from scripts import run_baseline_evaluation as eval_mod

	eval_mod.main(argv)


def run_analyze(argv: List[str]) -> None:
	from scripts import analyze_baseline_results as analyze_mod

	analyze_mod.main(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
	parser = argparse.ArgumentParser(description="TD-MPC2 baseline experiment driver.")
	subparsers = parser.add_subparsers(dest="command", required=True)

	# Train subcommand
	train_parser = subparsers.add_parser("train", help="Run Phase 1 training.")
	train_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
	train_parser.add_argument("--steps", type=int, default=500_000)
	train_parser.add_argument("--task", type=str, default="pendulum-swingup")
	train_parser.add_argument("--model-size", type=int, default=5, choices=[1, 5, 19, 48, 317])
	train_parser.add_argument("--exp-name", type=str, default="baseline")
	train_parser.add_argument("--results-dir", type=Path, default=BASELINE_RESULTS_DIR)
	train_parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
	train_parser.add_argument("--python", type=str, default=sys.executable)
	train_parser.add_argument("--extra-hydra-args", type=str, nargs="*")
	train_parser.add_argument("--save-video", action="store_true")
	train_parser.add_argument("--skip-existing", action="store_true")

	# Eval subcommand
	eval_parser = subparsers.add_parser("eval", help="Run Phase 2 zero-shot evaluation.")
	eval_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
	eval_parser.add_argument("--eval-episodes", type=int, default=100)
	eval_parser.add_argument("--task", type=str, default="pendulum-swingup-mass2")
	eval_parser.add_argument("--model-size", type=int, default=5, choices=[1, 5, 19, 48, 317])
	eval_parser.add_argument("--exp-name", type=str, default="baseline_mass2_eval")
	eval_parser.add_argument("--results-dir", type=Path, default=BASELINE_RESULTS_DIR)
	eval_parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
	eval_parser.add_argument("--save-video", action="store_true")
	eval_parser.add_argument("--plots", type=str, nargs="*")

	# Analyze subcommand
	analyze_parser = subparsers.add_parser("analyze", help="Aggregate logs and plots.")
	analyze_parser.add_argument("--results-dir", type=Path, default=BASELINE_RESULTS_DIR)
	analyze_parser.add_argument("--assets-dir", type=Path, default=BASELINE_ASSETS_DIR)
	analyze_parser.add_argument("--summary-path", type=Path, default=BASELINE_SUMMARY_PATH)

	# All subcommand
	all_parser = subparsers.add_parser("all", help="Run train + eval + analyze in sequence.")
	all_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
	all_parser.add_argument("--steps", type=int, default=500_000)
	all_parser.add_argument("--eval-episodes", type=int, default=100)
	all_parser.add_argument("--model-size", type=int, default=5, choices=[1, 5, 19, 48, 317])
	all_parser.add_argument("--train-exp-name", type=str, default="baseline")
	all_parser.add_argument("--eval-exp-name", type=str, default="baseline_mass2_eval")
	all_parser.add_argument("--results-dir", type=Path, default=BASELINE_RESULTS_DIR)
	all_parser.add_argument("--assets-dir", type=Path, default=BASELINE_ASSETS_DIR)
	all_parser.add_argument("--summary-path", type=Path, default=BASELINE_SUMMARY_PATH)
	all_parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)
	all_parser.add_argument("--extra-hydra-args", type=str, nargs="*")
	all_parser.add_argument("--train-save-video", action="store_true")
	all_parser.add_argument("--eval-save-video", action="store_true")
	all_parser.add_argument("--plots", type=str, nargs="*")
	all_parser.add_argument("--skip-existing", action="store_true")

	args = parser.parse_args(argv)

	if args.command == "train":
		run_train(build_train_argv(args))
	elif args.command == "eval":
		run_eval(build_eval_argv(args))
	elif args.command == "analyze":
		run_analyze(build_analyze_argv(args))
	elif args.command == "all":
		train_args = argparse.Namespace(
			seeds=args.seeds,
			steps=args.steps,
			task="pendulum-swingup",
			model_size=args.model_size,
			exp_name=args.train_exp_name,
			results_dir=args.results_dir,
			checkpoints_dir=args.checkpoints_dir,
			python=sys.executable,
			extra_hydra_args=args.extra_hydra_args,
			save_video=args.train_save_video,
			skip_existing=args.skip_existing,
		)
		run_train(build_train_argv(train_args))

		eval_args = argparse.Namespace(
			seeds=args.seeds,
			eval_episodes=args.eval_episodes,
			task="pendulum-swingup-mass2",
			model_size=args.model_size,
			exp_name=args.eval_exp_name,
			results_dir=args.results_dir,
			checkpoints_dir=args.checkpoints_dir,
			save_video=args.eval_save_video,
			plots=args.plots,
		)
		run_eval(build_eval_argv(eval_args))

		analyze_args = argparse.Namespace(
			results_dir=args.results_dir,
			assets_dir=args.assets_dir,
			summary_path=args.summary_path,
		)
		run_analyze(build_analyze_argv(analyze_args))


if __name__ == "__main__":
	main()
