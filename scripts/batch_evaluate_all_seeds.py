#!/usr/bin/env python
"""全シードのチェックポイントを一括でzero-shot評価"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="全シードのチェックポイントを一括評価")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints",
        help="チェックポイントディレクトリ")
    parser.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="pendulum_mass1_seed*.pt",
        help="チェックポイントのファイル名パターン")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "zero_shot_eval_batch",
        help="出力ディレクトリ")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="各環境での評価エピソード数")
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=3,
        help="動画保存するエピソード数")
    parser.add_argument(
        "--test-masses",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 2.5],
        help="テストする質量のリスト")
    parser.add_argument(
        "--model-size",
        type=int,
        default=5)
    parser.add_argument(
        "--compile",
        action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実行せずにコマンドを表示")
    
    args = parser.parse_args()
    
    # チェックポイントを検索
    checkpoints = sorted(args.checkpoints_dir.glob(args.checkpoint_pattern))
    
    if not checkpoints:
        print(f"Error: No checkpoints found matching '{args.checkpoint_pattern}' in {args.checkpoints_dir}")
        sys.exit(1)
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {cp.name}")
    print()
    
    # シード番号を抽出
    seed_checkpoints = []
    for cp in checkpoints:
        # seed番号を抽出（例: pendulum_mass1_seed0.pt -> 0）
        name = cp.stem
        if "seed" in name:
            try:
                seed = int(name.split("seed")[1])
                seed_checkpoints.append((seed, cp))
            except (IndexError, ValueError):
                print(f"Warning: Could not extract seed from {cp.name}, skipping")
                continue
    
    if not seed_checkpoints:
        print("Error: Could not extract seed numbers from checkpoint filenames")
        sys.exit(1)
    
    seed_checkpoints.sort()
    
    # 各チェックポイントを評価
    for seed, checkpoint in seed_checkpoints:
        print(f"\n{'='*70}")
        print(f"Evaluating seed {seed}: {checkpoint.name}")
        print(f"{'='*70}\n")
        
        # 出力ディレクトリをseed別に
        output_dir = args.output_dir / f"seed_{seed:02d}"
        
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_zero_shot.py"),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(output_dir),
            "--seed", str(seed),
            "--eval-episodes", str(args.eval_episodes),
            "--video-episodes", str(args.video_episodes),
            "--test-masses", *[str(m) for m in args.test_masses],
            "--model-size", str(args.model_size),
        ]
        
        if args.compile:
            cmd.append("--compile")
        
        if args.dry_run:
            print(f"[DRY RUN] Would execute:\n  {' '.join(cmd)}\n")
        else:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\nError evaluating seed {seed}: {e}")
                print("Continuing with next checkpoint...\n")
                continue
    
    if not args.dry_run:
        print(f"\n{'='*70}")
        print("✅ All evaluations complete!")
        print(f"{'='*70}")
        print(f"\nResults saved in: {args.output_dir}")
        print("\n各シードの結果:")
        for seed, _ in seed_checkpoints:
            seed_dir = args.output_dir / f"seed_{seed:02d}"
            if seed_dir.exists():
                # 最新のタイムスタンプディレクトリを探す
                timestamp_dirs = sorted(seed_dir.glob("*"))
                if timestamp_dirs:
                    latest = timestamp_dirs[-1]
                    report = latest / "report.html"
                    if report.exists():
                        print(f"  Seed {seed}: {report}")
    else:
        print("\n[DRY RUN] No commands were executed.")


if __name__ == "__main__":
    main()

