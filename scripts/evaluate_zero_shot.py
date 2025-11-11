#!/usr/bin/env python
"""Zero-shot評価スクリプト（動画付き比較版）

訓練環境（mass=1.0）とテスト環境（mass=2.0以上）で学習済みモデルを評価し、
動画やサマリーレポートを生成する。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import pathlib
from omegaconf import OmegaConf

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
from tdmpc2_transformer import TDMPC2Transformer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot評価（動画付き比較版）")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="学習済みチェックポイントのパス")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "zero_shot_eval",
        help="出力ディレクトリ")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="評価用のランダムシード")
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
        "--test-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5],
        help="テストするパラメータ倍率のリスト（1.0は訓練環境）")
    parser.add_argument(
        "--model-size",
        type=int,
        default=5,
        choices=[1, 5, 19, 48, 317])
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compileを有効化")
    return parser.parse_args()


DEFAULT_CKPT_FILENAMES = (
    "final.pt",
    "latest.pt",
    "model.pt",
    "checkpoint.pt",
    "agent.pt",
)


# タスクタイプごとのパラメータ設定マッピング
TASK_CONFIGS = {
    "pendulum-swingup": {
        "base_task": "pendulum-swingup",
        "param_name": "mass",
        "task_pattern": "pendulum-swingup-mass{multiplier}",
        "multiplier_to_suffix": {
            0.5: "",  # mass=0.5はまだ未実装、1.0をベースラインとする
            1.0: "",  # デフォルト
            1.5: "15",
            2.0: "2",
            2.5: "25",
            3.0: "30",
        }
    },
    "cup-catch": {
        "base_task": "cup-catch",
        "param_name": "ball_mass",
        "task_pattern": "cup-catch-ball-mass-{suffix}",
        "multiplier_to_suffix": {
            0.5: "05x",
            1.0: "10x",
            1.5: "15x",
            2.0: "20x",
            2.5: "25x",
        }
    },
    "hopper-stand": {
        "base_task": "hopper-stand",
        "param_name": "torso_mass",
        "task_pattern": "hopper-stand-torso-mass-{suffix}",
        "multiplier_to_suffix": {
            0.5: "05x",
            1.0: "10x",
            1.5: "15x",
            2.0: "20x",
            2.5: "25x",
        }
    },
    "reacher-three-easy": {
        "base_task": "reacher-three-easy",
        "param_name": "link_mass",
        "task_pattern": "reacher-three-easy-link-mass-{suffix}",
        "multiplier_to_suffix": {
            0.5: "05x",
            1.0: "10x",
            1.5: "15x",
            2.0: "20x",
            2.5: "25x",
        }
    },
}


def detect_task_type(checkpoint_cfg: Optional[dict], checkpoint_path: Path) -> str:
    """チェックポイントからタスクタイプを推定"""
    # cfgからタスク名を取得
    if checkpoint_cfg and "task" in checkpoint_cfg:
        task_name = checkpoint_cfg["task"]
        # -randomizedを除去してベースタスクを取得
        if task_name.endswith("-randomized"):
            task_name = task_name[:-11]
        
        # 既知のタスクタイプにマッチするか確認
        for task_type in TASK_CONFIGS.keys():
            if task_name == task_type:
                return task_type
        
        # 部分一致を試す
        for task_type in TASK_CONFIGS.keys():
            if task_type in task_name:
                return task_type
    
    # パスからも推定を試みる
    path_str = str(checkpoint_path).lower()
    for task_type in TASK_CONFIGS.keys():
        task_key = task_type.replace("-", "_")
        if task_key in path_str or task_type in path_str:
            return task_type
    
    # デフォルトはpendulum
    print(f"Warning: Could not detect task type from checkpoint, defaulting to pendulum-swingup")
    return "pendulum-swingup"


def generate_test_task_name(task_type: str, multiplier: float) -> Optional[str]:
    """倍率からテストタスク名を生成"""
    if task_type not in TASK_CONFIGS:
        return None
    
    config = TASK_CONFIGS[task_type]
    suffix_map = config["multiplier_to_suffix"]
    
    if multiplier not in suffix_map:
        print(f"Warning: multiplier {multiplier} not supported for {task_type}")
        return None
    
    suffix = suffix_map[multiplier]
    
    # pendulumの特殊ケース
    if task_type == "pendulum-swingup":
        if suffix == "":
            return "pendulum-swingup"
        else:
            return f"pendulum-swingup-mass{suffix}"
    
    # その他のタスク
    if suffix == "":
        return config["base_task"]
    else:
        return config["task_pattern"].format(suffix=suffix)


def resolve_checkpoint_path(path: Path) -> Path:
    """Return a concrete checkpoint file, resolving directories when needed."""
    if path.is_file():
        return path
    search_dirs = [path]
    model_dir = path / "models"
    if model_dir.is_dir():
        search_dirs.insert(0, model_dir)
    for directory in search_dirs:
        for name in DEFAULT_CKPT_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Could not find checkpoint file under {path}")


def load_checkpoint_metadata(path: Path):
    """Load minimal metadata (state + cfg) from a checkpoint file."""
    serialization = getattr(torch, "serialization", None)
    if serialization and hasattr(serialization, "add_safe_globals"):
        extra = [Path]
        posix_cls = getattr(pathlib, "PosixPath", None)
        if posix_cls:
            extra.append(posix_cls)
        serialization.add_safe_globals(extra)

    load_kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        state = torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        state = torch.load(path, **load_kwargs)
    except RuntimeError as exc:
        if "PosixPath" in str(exc) and serialization and hasattr(serialization, "safe_globals"):
            with serialization.safe_globals([Path, getattr(pathlib, "PosixPath", Path)]):
                state = torch.load(path, map_location="cpu", weights_only=False)
        else:
            raise
    cfg = None
    if isinstance(state, dict):
        cfg = state.get("cfg")
        if cfg is not None and not isinstance(cfg, dict):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    return state, cfg


def is_transformer_checkpoint(state_dict: dict, cfg_dict: Optional[dict]) -> bool:
    """Infer whether the checkpoint corresponds to the transformer agent."""
    if cfg_dict and cfg_dict.get("use_transformer"):
        return True
    model_state = state_dict.get("model") if isinstance(state_dict, dict) else None
    if not isinstance(model_state, dict):
        return False
    for key in model_state.keys():
        if key.startswith("state_emb.") or key.startswith("action_emb.") or key.startswith("pos_enc."):
            return True
    return False


def safe_close_env(env):
    """Attempt to close an environment and its wrapped children without raising."""
    visited = set()
    stack = [env]
    while stack:
        current = stack.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        close_fn = getattr(current, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except AttributeError:
                pass
        for attr in ("env", "unwrapped", "_env"):
            child = getattr(current, attr, None)
            if child is not None and id(child) not in visited:
                stack.append(child)


def build_cfg(
    checkpoint: Path,
    seed: int,
    task: str,
    model_size: int,
    compile_model: bool,
    base_cfg: Optional[dict] = None,
    use_transformer: bool = False,
):
    """設定を構築"""
    if base_cfg:
        cfg = OmegaConf.create(base_cfg)
    else:
        cfg = OmegaConf.load(PKG_ROOT / "config.yaml")
    cfg.task = task
    cfg.seed = seed
    if base_cfg and getattr(base_cfg, "get", None) and base_cfg.get("model_size") is not None:
        cfg.model_size = base_cfg.get("model_size")
    else:
        cfg.model_size = model_size
    cfg.checkpoint = str(checkpoint)
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.save_agent = False
    cfg.compile = compile_model
    cfg.eval_episodes = 1
    cfg.steps = 1
    cfg.exp_name = "zero_shot_eval"
    cfg.data_dir = str(REPO_ROOT / "datasets")
    cfg.use_transformer = bool(use_transformer)
    return parse_cfg(cfg)


def evaluate_env(
    agent: TDMPC2,
    env,
    n_episodes: int,
    save_video_count: int = 0,
    video_dir: Optional[Path] = None,
    multiplier: float = 1.0,
    param_name: str = "mass",
) -> dict:
    """1つの環境での評価を実行"""
    episode_returns = []
    episode_lengths = []
    videos = []
    
    for ep_idx in range(n_episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        t = 0
        
        # 動画保存が必要な場合
        frames = []
        if ep_idx < save_video_count and video_dir:
            frames.append(env.render())
        
        with torch.no_grad():
            while not done:
                action = agent.act(obs, t0=(t == 0), eval_mode=True)
                obs, reward, done, info = env.step(action)
                episode_return += float(reward)
                t += 1
                
                if frames is not None and len(frames) > 0:
                    frames.append(env.render())
        
        episode_returns.append(episode_return)
        episode_lengths.append(t)
        
        # 動画保存
        if frames:
            video_path = video_dir / f"{param_name}_{multiplier:.1f}x_ep_{ep_idx:03d}.mp4"
            save_video(frames, video_path)
            videos.append(str(video_path))
            print(f"  Episode {ep_idx}: return={episode_return:.2f}, len={t}, video={video_path.name}")
        else:
            print(f"  Episode {ep_idx}: return={episode_return:.2f}, len={t}")
    
    returns = np.array(episode_returns)
    lengths = np.array(episode_lengths)
    
    return {
        "multiplier": multiplier,
        "param_name": param_name,
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "median_return": float(np.median(returns)),
        "mean_length": float(np.mean(lengths)),
        "episode_returns": returns.tolist(),
        "episode_lengths": lengths.tolist(),
        "videos": videos,
    }


def save_video(frames: List[np.ndarray], path: Path):
    """動画をMP4で保存"""
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not installed, skipping video save")
        return
    
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_arr = np.asarray(frames, dtype=np.uint8)
    imageio.mimsave(path, frames_arr, fps=30)


def create_comparison_plots(results: List[dict], output_dir: Path, param_name: str = "mass"):
    """比較プロットを生成"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    multipliers = [r["multiplier"] for r in results]
    means = [r["mean_return"] for r in results]
    stds = [r["std_return"] for r in results]
    
    # ベースライン（1.0x）のインデックスを見つける
    baseline_idx = next((i for i, m in enumerate(multipliers) if m == 1.0), 0)
    
    # 1. 平均リターンの棒グラフ
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(multipliers)), means, yerr=stds, capsize=5)
    bars[baseline_idx].set_color('#2ca02c')  # 訓練環境を緑に
    for i in range(len(bars)):
        if i != baseline_idx:
            bars[i].set_color('#d62728')  # テスト環境を赤に
    
    ax.set_xlabel(f'{param_name.capitalize()} Multiplier', fontsize=12)
    ax.set_ylabel('Mean Episode Return', fontsize=12)
    ax.set_title(f'Zero-Shot Performance vs {param_name.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(multipliers)))
    ax.set_xticklabels([f"{m:.1f}x" for m in multipliers])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=means[baseline_idx], color='green', linestyle='--', alpha=0.5, label='Training performance')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_bar.png", dpi=150)
    plt.close()
    
    # 2. リターン分布のボックスプロット
    fig, ax = plt.subplots(figsize=(10, 6))
    returns_data = [r["episode_returns"] for r in results]
    bp = ax.boxplot(returns_data, labels=[f"{m:.1f}x" for m in multipliers], patch_artist=True)
    
    bp['boxes'][baseline_idx].set_facecolor('#2ca02c')
    for i in range(len(bp['boxes'])):
        if i != baseline_idx:
            bp['boxes'][i].set_facecolor('#d62728')
    
    ax.set_xlabel(f'{param_name.capitalize()} Multiplier', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title(f'Return Distribution Across Different {param_name.capitalize()}s', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_boxplot.png", dpi=150)
    plt.close()
    
    # 3. 性能低下率のプロット
    baseline_return = means[baseline_idx]
    degradation = [(baseline_return - m) / baseline_return * 100 for m in means]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(multipliers, degradation, 'o-', linewidth=2, markersize=8, color='#ff7f0e')
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax.fill_between(multipliers, 0, degradation, alpha=0.3, color='#ff7f0e')
    
    ax.set_xlabel(f'{param_name.capitalize()} Multiplier', fontsize=12)
    ax.set_ylabel('Performance Degradation (%)', fontsize=12)
    ax.set_title('Zero-Shot Performance Drop', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # データラベル追加
    for i, (m, d) in enumerate(zip(multipliers, degradation)):
        ax.text(m, d + 2, f"{d:.1f}%", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "degradation_curve.png", dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def create_html_report(results: List[dict], checkpoint: Path, output_dir: Path, eval_time: str, task_type: str = "unknown"):
    """HTMLレポートを生成"""
    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Zero-Shot Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 5px;
        }}
        .info-box {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-weight: bold;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #2c3e50;
        }}
        .metric-value.good {{
            color: #27ae60;
        }}
        .metric-value.bad {{
            color: #e74c3c;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .training-env {{
            background-color: #d4edda !important;
        }}
        .video-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .video-card {{
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .video-card h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        video {{
            width: 100%;
            border-radius: 4px;
        }}
        .plots img {{
            max-width: 100%;
            height: auto;
            margin: 10px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .plots {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>🎯 Zero-Shot Robustness Evaluation Report</h1>
    
    <div class="info-box">
        <h3>実験情報</h3>
        <div class="metric">
            <div class="metric-label">Checkpoint</div>
            <div class="metric-value">{checkpoint.name}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Task Type</div>
            <div class="metric-value">{task_type}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Evaluation Time</div>
            <div class="metric-value">{eval_time}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Environments</div>
            <div class="metric-value">{len(results)}</div>
        </div>
    </div>
    
    <h2>📊 Performance Summary</h2>
    <table>
        <thead>
            <tr>
                <th>{results[0]["param_name"].replace("_", " ").title()} Multiplier</th>
                <th>Mean Return</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>Max</th>
                <th>Median</th>
                <th>Episodes</th>
                <th>Degradation</th>
            </tr>
        </thead>
        <tbody>
"""
    
    # ベースライン（1.0x）を見つける
    baseline_idx = next((i for i, r in enumerate(results) if r["multiplier"] == 1.0), 0)
    baseline_return = results[baseline_idx]["mean_return"]
    
    for i, r in enumerate(results):
        degradation = (baseline_return - r["mean_return"]) / baseline_return * 100
        row_class = "training-env" if r["multiplier"] == 1.0 else ""
        degrade_class = "good" if degradation < 10 else ("bad" if degradation > 50 else "")
        
        html_content += f"""
            <tr class="{row_class}">
                <td><strong>{r["multiplier"]:.1f}x</strong>{'⭐' if r["multiplier"] == 1.0 else ''}</td>
                <td>{r["mean_return"]:.2f}</td>
                <td>{r["std_return"]:.2f}</td>
                <td>{r["min_return"]:.2f}</td>
                <td>{r["max_return"]:.2f}</td>
                <td>{r["median_return"]:.2f}</td>
                <td>{r["n_episodes"]}</td>
                <td class="metric-value {degrade_class}">{degradation:+.1f}%</td>
            </tr>
"""
    
    html_content += """
        </tbody>
    </table>
    
    <h2>📈 Comparison Plots</h2>
    <div class="plots">
        <img src="comparison_bar.png" alt="Bar Chart">
        <img src="comparison_boxplot.png" alt="Box Plot">
        <img src="degradation_curve.png" alt="Degradation Curve">
    </div>
    
    <h2>🎬 Evaluation Videos</h2>
"""
    
    for r in results:
        if r["videos"]:
            param_label = r["param_name"].replace("_", " ").title()
            is_baseline = r["multiplier"] == 1.0
            html_content += f"""
    <h3>{param_label} = {r["multiplier"]:.1f}x {'⭐ (Training Environment)' if is_baseline else ''}</h3>
    <div class="video-grid">
"""
            for video_path in r["videos"]:
                video_name = Path(video_path).name
                ep_num = video_name.split('_ep_')[1].split('.')[0]
                ep_idx = int(ep_num)
                ep_return = r["episode_returns"][ep_idx]
                
                html_content += f"""
        <div class="video-card">
            <h4>Episode {ep_idx} (Return: {ep_return:.2f})</h4>
            <video controls>
                <source src="videos/{video_name}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
"""
            html_content += "    </div>\n"
    
    html_content += """
</body>
</html>
"""
    
    report_path = output_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"\nHTML report saved to: {report_path}")


def main():
    args = parse_args()
    
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
    try:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    try:
        raw_state, checkpoint_cfg = load_checkpoint_metadata(checkpoint_path)
    except Exception as exc:
        print(f"Error: Failed to load checkpoint {checkpoint_path}: {exc}")
        sys.exit(1)
    if checkpoint_cfg is not None and not isinstance(checkpoint_cfg, dict):
        checkpoint_cfg = OmegaConf.to_container(checkpoint_cfg, resolve=True)
    use_transformer = is_transformer_checkpoint(raw_state, checkpoint_cfg)
    agent_cls = TDMPC2Transformer if use_transformer else TDMPC2
    agent_name = "Transformer TD-MPC2" if use_transformer else "TD-MPC2"
    del raw_state
    
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
    
    # タスクタイプを検出
    task_type = detect_task_type(checkpoint_cfg, checkpoint_path)
    param_name = TASK_CONFIGS[task_type]["param_name"]
    
    # 出力ディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / task_type.replace("-", "_") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Detected agent: {agent_name}")
    print(f"Detected task type: {task_type}")
    print(f"Test multipliers: {args.test_multipliers}")
    print(f"Episodes per environment: {args.eval_episodes}")
    print(f"Videos per environment: {args.video_episodes}")
    print()
    
    results = []
    
    for multiplier in args.test_multipliers:
        print(f"\n{'='*60}")
        print(f"Evaluating {param_name} = {multiplier:.1f}x")
        print(f"{'='*60}")
        
        # タスク名を生成
        task = generate_test_task_name(task_type, multiplier)
        if task is None:
            print(f"Warning: No task defined for {param_name}={multiplier}, skipping")
            continue
        
        # 設定構築
        cfg = build_cfg(
            checkpoint=checkpoint_path,
            seed=args.seed,
            task=task,
            model_size=args.model_size,
            compile_model=args.compile,
            base_cfg=checkpoint_cfg,
            use_transformer=use_transformer,
        )
        set_seed(cfg.seed)
        
        try:
            env = make_env(cfg)
        except Exception as e:
            print(f"Warning: Could not create task '{task}': {e}, skipping {param_name}={multiplier}")
            continue
        
        agent = agent_cls(cfg)
        agent.load(str(checkpoint_path))
        agent.eval()
        
        # 評価実行
        result = evaluate_env(
            agent=agent,
            env=env,
            n_episodes=args.eval_episodes,
            save_video_count=args.video_episodes,
            video_dir=video_dir,
            multiplier=multiplier,
            param_name=param_name,
        )
        results.append(result)
        safe_close_env(env)
        
        print(f"\nResults for {param_name}={multiplier:.1f}x:")
        print(f"  Mean return: {result['mean_return']:.2f} ± {result['std_return']:.2f}")
        print(f"  Range: [{result['min_return']:.2f}, {result['max_return']:.2f}]")
        print(f"  Median: {result['median_return']:.2f}")
    
    if not results:
        print("Error: No results collected")
        sys.exit(1)
    
    # 結果をJSON保存
    summary = {
        "checkpoint": str(checkpoint_path),
        "task_type": task_type,
        "param_name": param_name,
        "timestamp": timestamp,
        "seed": args.seed,
        "eval_episodes": args.eval_episodes,
        "video_episodes": args.video_episodes,
        "results": results,
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # CSVも保存
    with open(output_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "multiplier", "param_name", "mean_return", "std_return", "min_return", 
            "max_return", "median_return", "mean_length", "n_episodes"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "multiplier": r["multiplier"],
                "param_name": r["param_name"],
                "mean_return": r["mean_return"],
                "std_return": r["std_return"],
                "min_return": r["min_return"],
                "max_return": r["max_return"],
                "median_return": r["median_return"],
                "mean_length": r["mean_length"],
                "n_episodes": r["n_episodes"],
            })
    
    # プロット生成
    create_comparison_plots(results, output_dir, param_name)
    
    # HTMLレポート生成
    create_html_report(results, checkpoint_path, output_dir, timestamp, task_type)
    
    print(f"\n{'='*60}")
    print("✅ Evaluation complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"  - results.json")
    print(f"  - results.csv")
    print(f"  - report.html  <- ブラウザで開いて確認")
    print(f"  - videos/  ({sum(len(r['videos']) for r in results)} videos)")
    print(f"  - comparison_*.png  (3 plots)")


if __name__ == "__main__":
    main()

