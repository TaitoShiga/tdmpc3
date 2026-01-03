#!/usr/bin/env python3
"""Walker Walk training for actuator-randomized domain."""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ACTUATOR_SCALE_RANGE = (0.4, 1.4)

MODELS = [
    {
        "name": "DR",
        "task": "walker-walk_actuator_randomized",
        "exp_name": "walker_actuator_dr",
        "use_oracle": False,
        "use_model_c": False,
        "steps": 1000000,
    },
    {
        "name": "Model C",
        "task": "walker-walk_actuator_randomized",
        "exp_name": "walker_actuator_model_c",
        "use_oracle": False,
        "use_model_c": True,
        "steps": 1000000,
        "gru_pretrained": None,
    },
    {
        "name": "Oracle",
        "task": "walker-walk_actuator_randomized",
        "exp_name": "walker_actuator_oracle",
        "use_oracle": True,
        "use_model_c": False,
        "steps": 1000000,
    },
]

SEEDS = [0, 1, 2, 3, 4]
SAVE_VIDEO = False

C_PHYS_CONFIG = {
    "c_phys_dim": 1,
    "phys_param_type": "actuator",
    "phys_param_normalization": "standard",
    "context_length": 50,
    "gru_hidden_dim": 256,
}


def train_model(model_config, seed):
    name = model_config["name"]
    task = model_config["task"]
    exp_name = model_config["exp_name"]
    steps = model_config["steps"]

    print("\n" + "=" * 70)
    print(f"Training: {name} (seed={seed})")
    print(f"  Task: {task}")
    print(f"  Exp name: {exp_name}")
    print(f"  Steps: {steps}")
    print("=" * 70)

    cmd = [
        sys.executable,
        "tdmpc2/train.py",
        f"task={task}",
        f"exp_name={exp_name}",
        f"steps={steps}",
        f"seed={seed}",
        f"save_video={str(SAVE_VIDEO).lower()}",
        "enable_wandb=false",
        "compile=false",
    ]

    if model_config["use_oracle"]:
        cmd.append("use_oracle=true")
        cmd.extend([
            f"c_phys_dim={C_PHYS_CONFIG['c_phys_dim']}",
            f"phys_param_type={C_PHYS_CONFIG['phys_param_type']}",
            f"phys_param_normalization={C_PHYS_CONFIG['phys_param_normalization']}",
        ])

    if model_config["use_model_c"]:
        cmd.append("use_model_c=true")
        cmd.extend([
            f"c_phys_dim={C_PHYS_CONFIG['c_phys_dim']}",
            f"phys_param_type={C_PHYS_CONFIG['phys_param_type']}",
            f"phys_param_normalization={C_PHYS_CONFIG['phys_param_normalization']}",
            f"context_length={C_PHYS_CONFIG['context_length']}",
            f"gru_hidden_dim={C_PHYS_CONFIG['gru_hidden_dim']}",
        ])

        if model_config.get("gru_pretrained"):
            cmd.append(f"gru_pretrained={model_config['gru_pretrained']}")

    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")

    try:
        start = datetime.now()
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
        duration = datetime.now() - start
        print(f"Completed: {name} (seed={seed}) in {duration}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"Failed: {name} (seed={seed})")
        print(f"Error: {exc}")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted: {name} (seed={seed})")
        raise


def main():
    start_time = datetime.now()

    print("=" * 70)
    print("Walker Actuator Randomization Training")
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"\nModels: {len(MODELS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(MODELS) * len(SEEDS)}")
    print("\nModels:")
    for model in MODELS:
        print(f"  - {model['name']}: {model['task']}")
    print("\nPhysics param: actuator scale")
    print(f"  - DR range: {ACTUATOR_SCALE_RANGE[0]:.1f}x to {ACTUATOR_SCALE_RANGE[1]:.1f}x")
    print()

    os.chdir(REPO_ROOT)
    results = {}
    total_runs = len(MODELS) * len(SEEDS)
    current_run = 0

    for model in MODELS:
        model_name = model["name"]
        results[model_name] = {}

        for seed in SEEDS:
            current_run += 1
            print(f"\n{'#' * 70}")
            print(f"Progress {current_run}/{total_runs} - {model_name} seed={seed}")
            print(f"{'#' * 70}")

            try:
                success = train_model(model, seed)
                results[model_name][seed] = success
            except KeyboardInterrupt:
                print("\nTraining interrupted by user.")
                break

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 70)
    print("Training summary")
    print("=" * 70)
    for model_name, seed_results in results.items():
        successes = sum(1 for v in seed_results.values() if v)
        total = len(seed_results)
        print(f"{model_name}: {successes}/{total} succeeded")

    print(f"\nTotal time: {duration}")
    print(f"End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
