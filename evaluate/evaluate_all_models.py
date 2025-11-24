#!/usr/bin/env python
"""
4種類のモデル（baseline, dr, c, o）を評価し、results.csvを生成するスクリプト

使用方法:
    python evaluate/evaluate_all_models.py --seeds 0 1 2 3 4 --episodes 30 --test-params 0.5 1.0 1.5 2.0 2.5
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

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
from envs.wrappers.physics_param import wrap_with_physics_param
from tdmpc2 import TDMPC2
from tdmpc2_oracle import TDMPC2Oracle
from tdmpc2_model_c import TDMPC2ModelC


def parse_args():
    parser = argparse.ArgumentParser(description="全モデルの統一評価")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="評価するseedのリスト (default: 0 1 2 3 4)")
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="各(model, seed, param)あたりの評価エピソード数 (default: 30)")
    parser.add_argument(
        "--test-params",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5, 2.0, 2.5],
        help="テストする物理パラメータの倍率リスト (default: 0.5 1.0 1.5 2.0 2.5)")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results.csv",
        help="出力CSVファイルのパス (default: results.csv)")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=REPO_ROOT / "logs_remote",
        help="ログディレクトリ (default: logs_remote)")
    parser.add_argument(
        "--task",
        type=str,
        default="pendulum-swingup",
        help="タスク名 (default: pendulum-swingup)")
    parser.add_argument(
        "--model-size",
        type=int,
        default=5,
        choices=[1, 5, 19, 48, 317])
    return parser.parse_args()


def build_cfg(task: str, seed: int, model_size: int, use_oracle: bool = False, 
              use_model_c: bool = False) -> OmegaConf:
    """設定を構築"""
    cfg = OmegaConf.load(PKG_ROOT / "config.yaml")
    cfg.task = task
    cfg.seed = seed
    cfg.model_size = model_size
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.save_agent = False
    cfg.compile = False
    cfg.eval_episodes = 1
    cfg.steps = 1
    cfg.exp_name = "eval"
    cfg.data_dir = str(REPO_ROOT / "datasets")
    cfg.use_oracle = use_oracle
    cfg.use_model_c = use_model_c
    cfg.multitask = False
    cfg.obs = 'state'
    cfg.episodic = False
    
    # ??? を None に置き換え
    if cfg.get('checkpoint', '???') == '???':
        cfg.checkpoint = None
    if cfg.get('data_dir', '???') == '???':
        cfg.data_dir = None
    if cfg.get('gru_pretrained', '???') == '???':
        cfg.gru_pretrained = None
    
    cfg = parse_cfg(cfg)
    return cfg


def get_task_for_param(base_task: str, param_multiplier: float) -> str:
    """パラメータ倍率からタスク名を生成"""
    if base_task == "pendulum-swingup":
        if param_multiplier == 1.0:
            return "pendulum-swingup"
        elif param_multiplier == 0.5:
            return "pendulum-swingup-mass05"
        elif param_multiplier == 1.5:
            return "pendulum-swingup-mass15"
        elif param_multiplier == 2.0:
            return "pendulum-swingup-mass2"
        elif param_multiplier == 2.5:
            return "pendulum-swingup-mass25"
        elif param_multiplier == 3.0:
            return "pendulum-swingup-mass30"
        else:
            raise ValueError(f"Unsupported param_multiplier: {param_multiplier}")
    else:
        raise ValueError(f"Unsupported task: {base_task}")


def load_agent(checkpoint_path: Path, cfg: OmegaConf, use_oracle: bool, 
               use_model_c: bool):
    """エージェントをロード"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if use_model_c:
        agent = TDMPC2ModelC(cfg)
    elif use_oracle:
        agent = TDMPC2Oracle(cfg)
    else:
        agent = TDMPC2(cfg)
    
    agent.load(str(checkpoint_path))
    agent.eval()
    return agent


def evaluate_one_episode(agent, env, use_oracle: bool, use_model_c: bool, 
                        max_steps: int = 1000) -> dict:
    """1エピソードを評価"""
    obs = env.reset()
    done = False
    episode_return = 0.0
    t = 0
    
    # Model Cの場合は履歴をリセット
    if use_model_c:
        agent.reset_history()
    
    with torch.no_grad():
        while not done and t < max_steps:
            if use_model_c:
                action = agent.act(obs, t0=(t == 0), eval_mode=True)
                obs_next, reward, done, info = env.step(action)
                agent.update_history(obs, action)
                obs = obs_next
            elif use_oracle:
                c_phys = env.current_c_phys
                action = agent.act(obs, c_phys, t0=(t == 0), eval_mode=True)
                obs, reward, done, info = env.step(action)
            else:
                action = agent.act(obs, t0=(t == 0), eval_mode=True)
                obs, reward, done, info = env.step(action)
            
            episode_return += float(reward)
            t += 1
    
    return {
        "return": episode_return,
        "length": t,
        "success": float(info.get("success", 0.0))
    }


def evaluate_model(model_type: str, seed: int, param_multiplier: float, 
                  episodes: int, args: argparse.Namespace) -> List[dict]:
    """1つのモデル・seed・paramの組み合わせを評価"""
    # チェックポイントのパスを決定
    if model_type == "baseline":
        checkpoint_path = args.logs_dir / args.task / str(seed) / "baseline" / "models" / "final.pt"
        use_oracle = False
        use_model_c = False
    elif model_type == "dr":
        checkpoint_path = args.logs_dir / f"{args.task}-randomized" / str(seed) / "dr" / "models" / "final.pt"
        use_oracle = False
        use_model_c = False
    elif model_type == "c":
        checkpoint_path = args.logs_dir / f"{args.task}-randomized" / str(seed) / "modelc" / "models" / "final.pt"
        use_oracle = False
        use_model_c = True
    elif model_type == "o":
        checkpoint_path = args.logs_dir / f"{args.task}-randomized" / str(seed) / "oracle" / "models" / "final.pt"
        use_oracle = True
        use_model_c = False
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # タスク名を決定
    eval_task = get_task_for_param(args.task, param_multiplier)
    
    # 設定を構築
    cfg = build_cfg(
        task=eval_task,
        seed=seed,
        model_size=args.model_size,
        use_oracle=use_oracle,
        use_model_c=use_model_c
    )
    set_seed(cfg.seed)
    
    # 環境を作成
    env = make_env(cfg)
    if use_oracle or use_model_c:
        env = wrap_with_physics_param(env, cfg)
    
    # エージェントをロード
    try:
        agent = load_agent(checkpoint_path, cfg, use_oracle, use_model_c)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return []
    
    # 評価実行
    results = []
    for ep in range(episodes):
        result = evaluate_one_episode(agent, env, use_oracle, use_model_c)
        results.append({
            "model": model_type,
            "seed": seed,
            "param": param_multiplier,
            "episode": ep,
            "return": result["return"],
            "length": result["length"],
            "success": result["success"]
        })
    
    # 環境をクローズ
    try:
        env.close()
    except:
        pass
    
    return results


def main():
    args = parse_args()
    
    print("="*70)
    print("全モデル評価スクリプト")
    print("="*70)
    print(f"Seeds: {args.seeds}")
    print(f"Episodes per (model, seed, param): {args.episodes}")
    print(f"Test params: {args.test_params}")
    print(f"Output: {args.output}")
    print()
    
    all_results = []
    
    # 4種類のモデル × N seeds × M params
    models = ["baseline", "dr", "c", "o"]
    total_evals = len(models) * len(args.seeds) * len(args.test_params)
    
    with tqdm(total=total_evals, desc="Evaluating") as pbar:
        for model in models:
            for seed in args.seeds:
                for param in args.test_params:
                    desc = f"{model} seed={seed} param={param:.1f}"
                    pbar.set_description(desc)
                    
                    try:
                        results = evaluate_model(model, seed, param, args.episodes, args)
                        all_results.extend(results)
                        
                        if results:
                            mean_return = np.mean([r["return"] for r in results])
                            print(f"  {desc}: mean_return={mean_return:.2f}")
                    except Exception as e:
                        print(f"  Error in {desc}: {e}")
                    
                    pbar.update(1)
    
    # CSVに書き出し
    if all_results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            fieldnames = ["model", "seed", "param", "episode", "return", "length", "success"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow({k: row[k] for k in fieldnames})
        
        print()
        print("="*70)
        print(f"✓ 評価完了: {len(all_results)} 行を {args.output} に保存")
        print("="*70)
    else:
        print("Error: No results collected")
        sys.exit(1)


if __name__ == "__main__":
    main()

