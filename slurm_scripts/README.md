# tdmpc2 Slurm バッチスクリプト

## 概要

このディレクトリには、tdmpc2研究プロジェクトの4種類のモデル（Baseline, DR, Oracle, Model C）を、各5シード（合計20ジョブ）で訓練するためのSlurmバッチスクリプトが含まれています。

## ファイル構成

```
slurm_scripts/
├── README.md                  # このファイル
├── submit_all.sh              # 一括投入スクリプト
├── job_baseline_seed[0-4].sh  # Baseline（5ジョブ）
├── job_dr_seed[0-4].sh        # DR（5ジョブ）
├── job_oracle_seed[0-4].sh    # Oracle（5ジョブ）
└── job_modelc_seed[0-4].sh    # Model C（5ジョブ）
```

## 各モデルの説明

| モデル名 | 説明 | タスク | 追加引数 |
|---------|------|--------|---------|
| **Baseline** | 標準TD-MPC2（物理パラメータ固定） | `pendulum-swingup` | なし |
| **DR** | Domain Randomization | `pendulum-swingup-randomized` | なし |
| **Oracle** | 真の物理パラメータ使用（理論的上限） | `pendulum-swingup-randomized` | `use_oracle=true` |
| **Model C** | GRU推定器（提案手法） | `pendulum-swingup-randomized` | `use_model_c=true` |

すべてのジョブは **100,000ステップ** で訓練されます。

## 使用方法

### 1. スクリプトをクラスタにコピー

```bash
# ローカルマシンから
scp -r slurm_scripts/ shiga-t@10.232.11.170:~/tdmpc3/tdmpc3/
```

### 2. クラスタにログイン

```bash
ssh shiga-t@10.232.11.170
```

### 3. スクリプトディレクトリに移動

```bash
cd ~/tdmpc3/tdmpc3/slurm_scripts
```

### 4. 一括投入スクリプトに実行権限を付与

```bash
chmod +x submit_all.sh
chmod +x job_*.sh
```

### 5. すべてのジョブを投入

```bash
bash submit_all.sh
```

または、個別にジョブを投入:

```bash
sbatch job_baseline_seed0.sh
sbatch job_dr_seed1.sh
sbatch job_oracle_seed2.sh
sbatch job_modelc_seed3.sh
```

## ジョブの監視

### ジョブの状態を確認

```bash
squeue -u shiga-t
```

### 実行中のジョブのログをリアルタイムで確認

```bash
# 標準出力
tail -f ~/tdmpc3/tdmpc3/logs/tdmpc2-baseline-seed0-<jobid>.out

# 標準エラー
tail -f ~/tdmpc3/tdmpc3/logs/tdmpc2-baseline-seed0-<jobid>.err
```

### ジョブをキャンセル

```bash
# 特定のジョブ
scancel <jobid>

# すべてのジョブ
scancel -u shiga-t
```

## リソース設定

各ジョブのデフォルト設定:
- **GPU**: 1枚 (`--gres=gpu:1`)
- **CPU**: 8コア (`-c 8`)
- **メモリ**: 32GB (`--mem=32G`)
- **時間制限**: 24時間 (`-t 24:00:00`)

リソースを変更する場合は、各スクリプトの `#SBATCH` 行を編集してください。

## 出力ファイル

### 訓練ログ

- **標準出力**: `~/tdmpc3/tdmpc3/logs/tdmpc2-<model>-seed<N>-<jobid>.out`
- **標準エラー**: `~/tdmpc3/tdmpc3/logs/tdmpc2-<model>-seed<N>-<jobid>.err`

### 訓練結果

- **チェックポイント**: `~/tdmpc3/tdmpc3/checkpoints/` または `outputs/`
- **報酬の遷移**: tdmpc2が自動的に記録（TensorBoard互換形式）

## トラブルシューティング

### ジョブがキューに残ったまま実行されない

```bash
# ジョブの詳細を確認
scontrol show job <jobid>

# 利用可能なノードを確認
sinfo
```

### conda環境が見つからない

スクリプト内の以下の部分を確認:
```bash
source ~/.bashrc
conda activate tdmpc2
```

conda環境名が `tdmpc2` でない場合は、スクリプトを編集してください。

### GPU が見つからない

ログインノード (`mnode`) では GPU を使用できません。ジョブが計算ノードに割り当てられているか確認してください。

### メモリ不足エラー

スクリプトの `#SBATCH --mem=32G` を増やしてください（例: `--mem=64G`）。

## 実験完了後

### 結果の収集

```bash
# ローカルマシンに結果をダウンロード
scp -r shiga-t@10.232.11.170:~/tdmpc3/tdmpc3/outputs/ ./
scp -r shiga-t@10.232.11.170:~/tdmpc3/tdmpc3/logs/ ./
scp -r shiga-t@10.232.11.170:~/tdmpc3/tdmpc3/checkpoints/ ./
```

### 結果の分析

研究リポジトリに含まれる分析スクリプトを使用:
```bash
# 学習曲線のプロット
python analyze_results.py

# 推定精度の分析（Model C）
python analyze_estimation_accuracy.py

# 統計的検定
python compare_all_models_statistical.py
```

## サポート

問題が発生した場合は、`docs/slurm_cluster_prerequisites.md` を参照してください。

---

**作成日**: 2025年11月19日  
**対象プロジェクト**: tdmpc2 物理パラメータ推定研究

