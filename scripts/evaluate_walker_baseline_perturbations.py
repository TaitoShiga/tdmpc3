#!/usr/bin/env python3
"""Baseline Walker evaluation across multiple perturbations."""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TASK_GROUPS = {
    "mass": [
        ("walk_torso_mass_05x", 0.5),
        ("walk_torso_mass_10x", 1.0),
        ("walk_torso_mass_15x", 1.5),
        ("walk_torso_mass_20x", 2.0),
        ("walk_torso_mass_25x", 2.5),
        ("walk_torso_mass_03x", 0.3),
        ("walk_torso_mass_30x", 3.0),
        ("walk_torso_mass_35x", 3.5),
    ],
    "friction": [
        ("walk_friction_05x", 0.5),
        ("walk_friction_10x", 1.0),
        ("walk_friction_15x", 1.5),
        ("walk_friction_20x", 2.0),
    ],
    "actuator": [
        ("walk_actuator_06x", 0.6),
        ("walk_actuator_08x", 0.8),
        ("walk_actuator_10x", 1.0),
        ("walk_actuator_12x", 1.2),
        ("walk_actuator_14x", 1.4),
    ],
    "damping": [
        ("walk_damping_05x", 0.5),
        ("walk_damping_10x", 1.0),
        ("walk_damping_20x", 2.0),
    ],
    "gravity": [
        ("walk_gravity_08x", 0.8),
        ("walk_gravity_10x", 1.0),
        ("walk_gravity_12x", 1.2),
    ],
}

RESULT_LINE_RE = re.compile(r"R:\s*([-\d\.]+)\s+S:\s*([-\d\.]+)")


def parse_result(output: str):
    reward = None
    success = None
    for line in output.splitlines():
        match = RESULT_LINE_RE.search(line)
        if match:
            reward = float(match.group(1))
            success = float(match.group(2))
    return reward, success


def build_task_list(only_groups):
    if not only_groups:
        only_groups = list(TASK_GROUPS.keys())
    tasks = []
    for group in only_groups:
        for task_name, scale in TASK_GROUPS[group]:
            tasks.append((group, task_name, scale))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline Walker across perturbations.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "logs" / "walker-walk" / "0" / "walker_baseline" / "models" / "final.pt",
        help="Path to baseline checkpoint.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=30,
        help="Number of episodes per task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "walker_baseline_perturbations.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(TASK_GROUPS.keys()),
        default=None,
        help="Subset of perturbation groups to evaluate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.exists():
        print(f"Error: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    tasks = build_task_list(args.only)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")

    for idx, (group, task_name, scale) in enumerate(tasks, start=1):
        cmd = [
            sys.executable,
            "tdmpc2/evaluate.py",
            f"task=walker-{task_name}",
            f"checkpoint={checkpoint}",
            f"eval_episodes={args.eval_episodes}",
            f"seed={args.seed}",
            "save_video=false",
            "compile=false",
        ]
        print(f"[{idx}/{len(tasks)}] {group}: {task_name} (scale={scale})")
        if args.dry_run:
            print("  DRY RUN:", " ".join(cmd))
            continue

        try:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            reward, success = parse_result(result.stdout)
            results.append((group, task_name, scale, reward, success, "ok"))
        except subprocess.CalledProcessError as exc:
            results.append((group, task_name, scale, None, None, "failed"))
            print("  Error:", exc, file=sys.stderr)
            if exc.stdout:
                print(exc.stdout, file=sys.stderr)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)

    if args.dry_run:
        return

    with args.output.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["group", "task", "scale", "reward", "success", "status"])
        writer.writerows(results)

    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
