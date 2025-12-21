#!/usr/bin/env python3
"""Cheetah評価のテストスクリプト

チェックポイントの存在確認と1つの評価を実行してテストする。
"""

import sys
from pathlib import Path

# パス確認
REPO_ROOT = Path.home() / "tdmpc3" / "tdmpc3"

print("="*70)
print("Cheetah Checkpoint Test")
print("="*70)
print(f"Repository root: {REPO_ROOT}")
print()

# チェックポイントパスを確認
models = [
    ("Baseline", "cheetah-run_friction04", "cheetah_baseline"),
    ("DR", "cheetah-run_randomized", "cheetah_dr"),
    ("Model C", "cheetah-run_randomized", "cheetah_modelc"),
    ("Oracle", "cheetah-run_randomized", "cheetah_oracle"),
]

print("Checking checkpoints for seed 0:")
print("-" * 70)

found_checkpoints = []

for name, task, exp_name in models:
    checkpoint_path = REPO_ROOT / "logs" / task / "0" / exp_name / "models" / "final.pt"
    exists = checkpoint_path.exists()
    status = "✅" if exists else "❌"
    
    print(f"{status} {name:12} {checkpoint_path}")
    
    if exists:
        found_checkpoints.append((name, task, exp_name, checkpoint_path))

print()
print("="*70)

if not found_checkpoints:
    print("❌ No checkpoints found!")
    print("\nMake sure training is complete and checkpoints are saved.")
    sys.exit(1)

# 最初に見つかったチェックポイントで評価テスト
print(f"Found {len(found_checkpoints)} checkpoint(s)")
print()

test_name, test_task, test_exp, test_checkpoint = found_checkpoints[0]

print(f"Testing evaluation with: {test_name}")
print(f"Checkpoint: {test_checkpoint}")
print(f"Evaluating on: cheetah-run_friction04 (default friction)")
print()

# 評価コマンドを表示
eval_cmd = f"""
python tdmpc2/evaluate.py \\
    task=cheetah-run_friction04 \\
    model={test_checkpoint} \\
    episodes=3 \\
    seed=0 \\
    save_video=false
"""

print("Evaluation command:")
print(eval_cmd)
print()

# 実際に評価を実行するか確認
response = input("Run evaluation test? (y/n): ")

if response.lower() == 'y':
    import subprocess
    
    cmd = [
        sys.executable,
        'tdmpc2/evaluate.py',
        'task=cheetah-run_friction04',
        f'model={test_checkpoint}',
        'episodes=3',
        'seed=0',
        'save_video=false',
    ]
    
    print("\nRunning evaluation...")
    print("-" * 70)
    
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    
    if result.returncode == 0:
        print("-" * 70)
        print("✅ Evaluation test successful!")
        print()
        print("Next steps:")
        print("  1. Run full evaluation:")
        print("     python scripts/evaluate_cheetah_all_models.py")
        print()
        print("  2. Or submit to Slurm:")
        print("     sbatch slurm_scripts/evaluate_cheetah_all.sh")
    else:
        print("-" * 70)
        print("❌ Evaluation test failed!")
        print("Check error messages above.")
else:
    print("\nTest skipped.")
    print("\nTo run evaluation manually:")
    print(eval_cmd)

print("="*70)

