#!/usr/bin/env python
"""全タスクでzero-shot評価を一括実行"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "logs"

# 評価するチェックポイントの設定
# (タスクディレクトリ名, サブディレクトリ, 説明)
CHECKPOINTS_TO_EVAL = [
    # Baseline checkpoints
    ("pendulum-swingup", "0/baseline", "Pendulum Baseline"),
    ("cup-catch", "0/ball_in_cup_baseline", "Cup-Catch Baseline"),
    ("hopper-stand", "0/hopper_baseline", "Hopper-Stand Baseline"),
    ("reacher-three-easy", "0/reacher_baseline", "Reacher Baseline"),
    
    # DR checkpoints
    ("pendulum-swingup-randomized", "0/default", "Pendulum DR"),
    ("cup-catch-randomized", "0/ball_in_cup_dr", "Cup-Catch DR"),
    ("hopper-stand-randomized", "0/hopper_dr", "Hopper-Stand DR"),
    ("reacher-three-easy-randomized", "0/reacher_dr", "Reacher DR"),
]

# 評価設定
EVAL_EPISODES = 10
VIDEO_EPISODES = 3
MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 2.5]


def find_checkpoint(task_dir: str, subdir: str) -> Path | None:
    """チェックポイントファイルを探す"""
    checkpoint_dir = LOGS_DIR / task_dir / subdir / "models"
    
    if not checkpoint_dir.exists():
        print(f"  ✗ Directory not found: {checkpoint_dir}")
        return None
    
    # final.pt を優先的に探す
    for checkpoint_name in ["final.pt", "latest.pt", "model.pt"]:
        checkpoint = checkpoint_dir / checkpoint_name
        if checkpoint.exists():
            return checkpoint
    
    print(f"  ✗ No checkpoint found in: {checkpoint_dir}")
    return None


def run_zero_shot_eval(checkpoint: Path, description: str) -> bool:
    """Zero-shot評価を実行"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {description}")
    print(f"Checkpoint: {checkpoint}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_zero_shot.py"),
        "--checkpoint", str(checkpoint),
        "--eval-episodes", str(EVAL_EPISODES),
        "--video-episodes", str(VIDEO_EPISODES),
        "--test-multipliers", *[str(m) for m in MULTIPLIERS],
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {description}: Success!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description}: Failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ {description}: Interrupted by user")
        raise


def main():
    print("="*80)
    print("Batch Zero-Shot Evaluation")
    print("="*80)
    print(f"Eval episodes: {EVAL_EPISODES}")
    print(f"Video episodes: {VIDEO_EPISODES}")
    print(f"Test multipliers: {MULTIPLIERS}")
    print(f"Total checkpoints to evaluate: {len(CHECKPOINTS_TO_EVAL)}")
    print("="*80)
    
    results = []
    
    for task_dir, subdir, description in CHECKPOINTS_TO_EVAL:
        checkpoint = find_checkpoint(task_dir, subdir)
        
        if checkpoint is None:
            print(f"⏭️  Skipping {description}: checkpoint not found\n")
            results.append((description, "skipped", None))
            continue
        
        try:
            success = run_zero_shot_eval(checkpoint, description)
            results.append((description, "success" if success else "failed", checkpoint))
        except KeyboardInterrupt:
            print("\n\n⚠️ Batch evaluation interrupted by user")
            break
    
    # サマリー表示
    print("\n\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for description, status, checkpoint in results:
        status_emoji = {
            "success": "✅",
            "failed": "❌",
            "skipped": "⏭️"
        }[status]
        print(f"{status_emoji} {description}: {status}")
    
    success_count = sum(1 for _, s, _ in results if s == "success")
    failed_count = sum(1 for _, s, _ in results if s == "failed")
    skipped_count = sum(1 for _, s, _ in results if s == "skipped")
    
    print(f"\nTotal: {len(results)} | Success: {success_count} | Failed: {failed_count} | Skipped: {skipped_count}")
    print("="*80)
    
    # 結果の場所を表示
    output_dir = REPO_ROOT / "outputs" / "zero_shot_eval"
    if output_dir.exists():
        print(f"\n📂 Results saved in: {output_dir}")
        print("   Open report.html files in browser to view detailed results")


if __name__ == "__main__":
    main()

