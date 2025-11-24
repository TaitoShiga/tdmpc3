# 評価ジョブ投入ガイド

Slurmクラスタで評価を実行するためのスクリプト群。

## ファイル一覧

```
slurm_scripts/
├── job_eval_seed0.sh         # Seed 0 の評価ジョブ
├── job_eval_seed1.sh         # Seed 1 の評価ジョブ
├── job_eval_seed2.sh         # Seed 2 の評価ジョブ
├── job_eval_seed3.sh         # Seed 3 の評価ジョブ
├── job_eval_seed4.sh         # Seed 4 の評価ジョブ
├── job_test_eval.sh          # テスト評価ジョブ
├── submit_eval_all.sh        # 一括投入スクリプト
├── merge_eval_results.sh     # 結果マージスクリプト
└── README_EVAL.md            # このファイル
```

## 使用方法

### ステップ 0: 事前準備

作業ディレクトリのパスを確認・修正：

```bash
# 各 job_eval_seed*.sh の cd コマンドを自分の環境に合わせて変更
cd ~/tdmpc3/tdmpc3  # <- ここを修正
```

### ステップ 1: テスト実行（推奨）

まず軽量なテストジョブで動作確認：

```bash
sbatch slurm_scripts/job_test_eval.sh
```

**内容**: 1 seed × 1 param × 5 episodes（約5-10分）

**確認**:
```bash
# ジョブの状態確認
squeue -u $USER

# ログ確認
tail -f logs/tdmpc2-test-eval-<jobid>.out

# 結果確認
cat results_test.csv
```

### ステップ 2: 本番実行

テストが成功したら、5つの評価ジョブを一括投入：

```bash
bash slurm_scripts/submit_eval_all.sh
```

**内容**: 各ジョブで 4 models × 5 params × 30 episodes（1ジョブ約2-4時間）

**投入されるジョブ**:
- `job_eval_seed0.sh` → `results_seed0.csv`
- `job_eval_seed1.sh` → `results_seed1.csv`
- `job_eval_seed2.sh` → `results_seed2.csv`
- `job_eval_seed3.sh` → `results_seed3.csv`
- `job_eval_seed4.sh` → `results_seed4.csv`

### ステップ 3: ジョブ監視

```bash
# すべてのジョブの状態確認
squeue -u $USER

# 特定のジョブのログを監視
tail -f logs/tdmpc2-eval-seed0-<jobid>.out
tail -f logs/tdmpc2-eval-seed1-<jobid>.out
# ...

# ジョブの詳細情報
scontrol show job <jobid>

# ジョブのキャンセル（必要な場合）
scancel <jobid>

# すべての評価ジョブをキャンセル（必要な場合）
scancel -u $USER -n tdmpc2-eval-seed*
```

### ステップ 4: 結果のマージ

すべてのジョブが完了したら、結果を1つのファイルにマージ：

```bash
bash slurm_scripts/merge_eval_results.sh
```

**出力**: `results.csv`（約3,600行のデータ）

**確認**:
```bash
# 行数確認（ヘッダー + 3600行程度のはず）
wc -l results.csv

# サンプル確認
head -20 results.csv

# 各モデルのデータ数確認
cut -d',' -f1 results.csv | sort | uniq -c
```

### ステップ 5: 統計解析

マージされた `results.csv` を解析：

```bash
python evaluate/analyze_results.py
```

**出力**:
- コンソール出力: 統計サマリー
- `fig_per_param.png`: param別の性能比較
- `fig_overall.png`: 全体性能比較

## ジョブの仕様

### リソース設定

```bash
#SBATCH --gres=gpu:1    # GPU 1枚
#SBATCH -c 8            # CPU 8コア
#SBATCH --mem=32G       # メモリ 32GB
#SBATCH -t 12:00:00     # 制限時間 12時間
```

### 評価内容（1ジョブあたり）

- **モデル数**: 4（baseline, dr, Model C, Oracle）
- **パラメータ**: 5（mass multiplier: 0.5, 1.0, 1.5, 2.0, 2.5）
- **エピソード**: 30
- **合計**: 4 × 5 × 30 = 600 episodes

### 並列実行

5つのジョブは独立しているため、並列実行可能。クラスタのリソースが許す限り同時実行される。

## トラブルシューティング

### ジョブが PENDING のまま

```bash
# 理由を確認
squeue -u $USER --start

# クラスタの状況確認
sinfo
```

**原因**: GPUが不足している可能性

**対処**: 時間帯をずらす、または一部のジョブを後で投入

### ジョブが FAILED

```bash
# エラーログ確認
cat logs/tdmpc2-eval-seed<N>-<jobid>.err

# 標準出力確認
cat logs/tdmpc2-eval-seed<N>-<jobid>.out
```

**よくあるエラー**:
1. **conda環境が見つからない**
   → `eval "$(conda shell.bash hook)"` が正しく実行されているか確認

2. **チェックポイントが見つからない**
   → `logs_remote` ディレクトリのパスを確認

3. **CUDA out of memory**
   → `--episodes` を減らす（30 → 20 → 10）

### 一部のジョブだけ失敗

個別に再実行：

```bash
# 例: seed 2 だけ再実行
sbatch slurm_scripts/job_eval_seed2.sh
```

### 結果ファイルが空

```bash
# ジョブが完了しているか確認
squeue -u $USER

# 出力ファイルの確認
ls -lh results_seed*.csv

# 内容確認
wc -l results_seed*.csv
```

## カスタマイズ

### 異なるエピソード数

各 `job_eval_seed*.sh` の `--episodes` を変更：

```bash
--episodes 20  # 30 → 20
```

### 異なるパラメータ

各 `job_eval_seed*.sh` の `--test-params` を変更：

```bash
--test-params 1.0 2.0 3.0  # 3つのパラメータのみ
```

### リソース調整

メモリやCPUを調整：

```bash
#SBATCH --mem=64G       # 32GB → 64GB
#SBATCH -c 16           # 8コア → 16コア
#SBATCH -t 24:00:00     # 12時間 → 24時間
```

## 参考

- [Slurm公式ドキュメント](https://slurm.schedmd.com/)
- [評価パイプライン全体のドキュメント](../evaluate/README.md)
- [使用例とトラブルシューティング](../evaluate/USAGE_EXAMPLE.md)

