# Context Length Ablation - 実行ガイド

## 📦 作成されたファイル

### Slurmジョブスクリプト（15個）

#### Context Length 10
- `slurm_scripts/job_ctx10_seed0.sh`
- `slurm_scripts/job_ctx10_seed1.sh`
- `slurm_scripts/job_ctx10_seed2.sh`

#### Context Length 25
- `slurm_scripts/job_ctx25_seed0.sh`
- `slurm_scripts/job_ctx25_seed1.sh`
- `slurm_scripts/job_ctx25_seed2.sh`

#### Context Length 50
- `slurm_scripts/job_ctx50_seed0.sh`
- `slurm_scripts/job_ctx50_seed1.sh`
- `slurm_scripts/job_ctx50_seed2.sh`

#### Context Length 100
- `slurm_scripts/job_ctx100_seed0.sh`
- `slurm_scripts/job_ctx100_seed1.sh`
- `slurm_scripts/job_ctx100_seed2.sh`

#### Context Length 200
- `slurm_scripts/job_ctx200_seed0.sh`
- `slurm_scripts/job_ctx200_seed1.sh`
- `slurm_scripts/job_ctx200_seed2.sh`

### 一括投入スクリプト

#### 全部投入（15ジョブ）
- `slurm_scripts/submit_all_context_lengths_3seeds.sh`

#### 1 seedだけ投入（5ジョブ）
- `slurm_scripts/submit_all_context_lengths.sh`

#### Context Length別に全seeds投入（3ジョブ × 5）
- `slurm_scripts/submit_ctx10_all_seeds.sh`
- `slurm_scripts/submit_ctx25_all_seeds.sh`
- `slurm_scripts/submit_ctx50_all_seeds.sh`
- `slurm_scripts/submit_ctx100_all_seeds.sh`
- `slurm_scripts/submit_ctx200_all_seeds.sh`

---

## 🚀 実行方法

### パターン1: 全部一気に投入（推奨、GPU 15台）

```bash
bash slurm_scripts/submit_all_context_lengths_3seeds.sh
```

**投入されるジョブ**:
- 5 context lengths × 3 seeds = **15ジョブ**
- 全て並列実行
- 所要時間: 約12-18時間

---

### パターン2: 1 seedだけ投入（GPU 5台）

```bash
bash slurm_scripts/submit_all_context_lengths.sh
```

**投入されるジョブ**:
- 5 context lengths × 1 seed (seed=0) = **5ジョブ**
- 全て並列実行
- 所要時間: 約12-18時間

---

### パターン3: Context Length別に投入

#### 例: Context Length 10だけ（GPU 3台）

```bash
bash slurm_scripts/submit_ctx10_all_seeds.sh
```

**投入されるジョブ**:
- 1 context length × 3 seeds = **3ジョブ**
- 並列実行

#### 順次投入

```bash
# まずctx=10から試す
bash slurm_scripts/submit_ctx10_all_seeds.sh

# 結果を見て次を決める
bash slurm_scripts/submit_ctx50_all_seeds.sh
bash slurm_scripts/submit_ctx100_all_seeds.sh
```

---

### パターン4: 個別にジョブ投入

```bash
# 1つだけテスト
sbatch slurm_scripts/job_ctx10_seed0.sh

# 複数を手動で投入
sbatch slurm_scripts/job_ctx10_seed0.sh
sbatch slurm_scripts/job_ctx25_seed0.sh
sbatch slurm_scripts/job_ctx50_seed0.sh
```

---

## 📊 ジョブ確認

### 投入されたジョブを確認

```bash
squeue -u $USER
```

### ログをリアルタイム確認

```bash
# すべてのログ
tail -f logs/tdmpc2-ctx*-*.out

# 特定のcontext length
tail -f logs/tdmpc2-ctx10-*.out

# 特定のseed
tail -f logs/tdmpc2-ctx*-seed0-*.out
```

### ジョブをキャンセル

```bash
# 特定のジョブ
scancel <JOB_ID>

# 自分の全ジョブ
scancel -u $USER

# 特定のパターンのジョブ
scancel --name=tdmpc2-ctx10*
```

---

## 📂 出力ファイル

学習完了後、以下の場所にチェックポイントとログが保存される：

```
logs/pendulum-swingup-randomized/
  0/                           # seed 0
    modelc_ctx10/
      models/
        final.pt              # チェックポイント
      eval.csv                # 学習曲線データ
      eval_video/
        *.mp4
    modelc_ctx25/
      models/final.pt
      eval.csv
    modelc_ctx50/
      models/final.pt
      eval.csv
    modelc_ctx100/
      models/final.pt
      eval.csv
    modelc_ctx200/
      models/final.pt
      eval.csv
  1/                           # seed 1
    modelc_ctx10/
      ...
    modelc_ctx25/
      ...
    ...
  2/                           # seed 2
    modelc_ctx10/
      ...
    ...
```

---

## 📈 結果の分析

### 学習曲線の可視化

```python
import pandas as pd
import matplotlib.pyplot as plt

context_lengths = [10, 25, 50, 100, 200]
seed = 0

fig, ax = plt.subplots(figsize=(10, 6))

for ctx_len in context_lengths:
    path = f"logs/pendulum-swingup-randomized/{seed}/modelc_ctx{ctx_len}/eval.csv"
    df = pd.read_csv(path)
    ax.plot(df["step"], df["episode_reward"], label=f"ctx={ctx_len}")

ax.legend()
ax.set_xlabel("Steps")
ax.set_ylabel("Episode Return")
ax.set_title("Learning Curves: Context Length Ablation")
ax.grid(alpha=0.3)
plt.savefig("context_ablation_learning_curves.png", dpi=150)
plt.show()
```

### 最終性能の比較

```python
import pandas as pd
import numpy as np

context_lengths = [10, 25, 50, 100, 200]
seeds = [0, 1, 2]

results = {}

for ctx_len in context_lengths:
    returns = []
    for seed in seeds:
        path = f"logs/pendulum-swingup-randomized/{seed}/modelc_ctx{ctx_len}/eval.csv"
        df = pd.read_csv(path)
        # 最後10%の平均
        final_return = df["episode_reward"].tail(int(len(df) * 0.1)).mean()
        returns.append(final_return)
    
    results[ctx_len] = {
        "mean": np.mean(returns),
        "std": np.std(returns)
    }

for ctx_len, stats in results.items():
    print(f"ctx={ctx_len:3d}: {stats['mean']:.1f} ± {stats['std']:.1f}")
```

---

## ⏱️ 実行時間と計算コスト

| 実行パターン | ジョブ数 | GPU数 | 並列時間 | 逐次時間 |
|------------|---------|-------|---------|---------|
| 全部（3 seeds） | 15 | 15 | 12-18h | 180-270h |
| 1 seed | 5 | 5 | 12-18h | 60-90h |
| 1 context length | 3 | 3 | 12-18h | 36-54h |
| 単一ジョブ | 1 | 1 | 12-18h | 12-18h |

---

## ✅ 実行チェックリスト

### 実行前

- [ ] Slurm環境にアクセス可能
- [ ] GPU割り当て確認（必要数の確保）
- [ ] Conda環境 `tdmpc2` がアクティブ
- [ ] 作業ディレクトリのパス確認（スクリプト内の`cd`コマンド）

### 実行中

- [ ] ジョブが投入されたか確認（`squeue`）
- [ ] ログが出力されているか確認（`tail -f`）
- [ ] エラーがないか確認

### 実行後

- [ ] 全ジョブが完了したか確認
- [ ] チェックポイントが保存されたか確認
- [ ] eval.csvが生成されたか確認
- [ ] 学習曲線を可視化
- [ ] 最終性能を比較

---

## 🐛 トラブルシューティング

### ジョブが投入されない

```bash
# スクリプトに実行権限を付与
chmod +x slurm_scripts/*.sh

# パスを確認
ls -la slurm_scripts/
```

### GPUが足りない

```bash
# 利用可能なGPUを確認
sinfo -o "%20N %10c %10m %25f %10G"

# ジョブを少しずつ投入
sbatch slurm_scripts/job_ctx10_seed0.sh
# 完了を待ってから次を投入
```

### メモリ不足

スクリプト内の`--mem=32G`を`--mem=64G`に変更

### Context Length 200が遅い

正常。長いcontext lengthはメモリと計算量が増える。

---

## 💡 推奨実行戦略

### 戦略1: まず1 seedで全context lengthを試す

```bash
bash slurm_scripts/submit_all_context_lengths.sh
```

→ 結果を見て傾向を確認  
→ 有望なcontext lengthで3 seeds実行

### 戦略2: 段階的に実行

```bash
# ステップ1: ctx=10, 50, 100 だけ試す（seed 0）
sbatch slurm_scripts/job_ctx10_seed0.sh
sbatch slurm_scripts/job_ctx50_seed0.sh
sbatch slurm_scripts/job_ctx100_seed0.sh

# ステップ2: 結果を見て追加
# もし100が良ければ
bash slurm_scripts/submit_ctx100_all_seeds.sh
```

### 戦略3: 一気に全部（GPU豊富な場合）

```bash
bash slurm_scripts/submit_all_context_lengths_3seeds.sh
```

---

## 📝 次のステップ

1. ✅ スクリプト作成完了
2. [ ] ジョブ投入
3. [ ] 学習完了待ち（12-18時間）
4. [ ] 結果の可視化
5. [ ] 最適なcontext lengthの特定
6. [ ] Zero-Shot評価
7. [ ] 論文に結果を追加


