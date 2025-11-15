# Oracle Mode 使用ガイド

## 🎯 概要

**Oracle Mode** は、真の物理パラメータを常にプランナーに注入することで、物理推定が完璧な場合の理論的上限を検証するモードです。

コマンドライン引数 `use_oracle=true` で有効化できます。

---

## ✅ 確認事項

### 1. 評価について
- **最低限必要なのは `.pt` ファイル（チェックポイント）のみ**
- 評価は後回しでOK
- 学習時に自動的に `logs/` ディレクトリに保存される

### 2. コマンドライン引数での切り替え
- ✅ **対応しました！** `use_oracle=true/false` で切り替え可能
- デフォルトは `use_oracle=false`（標準モード）

### 3. Domain Randomization (DR) との統合
- ✅ **DRの一様分布から取得した値を使用できます**
- `PhysicsParamWrapper` が `physics.model.body_mass[-1]` から真の値を取得
- DRで設定された質量（`uniform(0.5, 2.5)`）を正しく読み取れる

### 4. 物理パラメータの扱い
- ✅ **エピソードごとに物理パラメータは変わります（DRの場合）**
- ✅ **エピソード中は固定です**
- つまり：
  - **DRなし（`pendulum-swingup`）:** 全エピソードで質量=1.0
  - **DRあり（`pendulum-swingup-randomized`）:** エピソードごとに `uniform(0.5, 2.5)` からサンプル
  - **Oracle:** 上記の「その時点の真の質量」をプランナーに注入

---

## 🚀 実行方法

### 基本コマンド

```bash
cd tdmpc2

# 標準モード（Oracleなし）
python train.py task=pendulum-swingup seed=0

# Oracleモード
python train.py task=pendulum-swingup use_oracle=true seed=0
```

### 4つの実験パターン

#### 1. ベースライン（DRなし）
```bash
python train.py \
    task=pendulum-swingup \
    seed=0 \
    steps=500000
```
- **説明:** 標準的なTD-MPC2、質量=1.0で固定
- **用途:** 基本性能のベンチマーク

#### 2. ベースライン（DRあり）
```bash
python train.py \
    task=pendulum-swingup-randomized \
    seed=0 \
    steps=500000
```
- **説明:** Domain Randomization、エピソードごとに質量が変動
- **用途:** 汎化性能の評価

#### 3. Oracle（DRなし）← **理論的上限**
```bash
python train.py \
    task=pendulum-swingup \
    use_oracle=true \
    seed=0 \
    steps=500000
```
- **説明:** 完璧な物理情報（質量=1.0）をプランナーに注入
- **用途:** 固定環境での理論的上限

#### 4. Oracle（DRあり）← **最も重要！**
```bash
python train.py \
    task=pendulum-swingup-randomized \
    use_oracle=true \
    seed=0 \
    steps=500000
```
- **説明:** 完璧な物理情報をプランナーに注入、DRで汎化性能を評価
- **用途:** 変動環境での理論的上限（Model Cの目標）

---

## 🔬 実験デザイン

### 推奨する実験の順序

```
ステップ1: 動作確認
  python test_oracle_with_dr.py
  → DRとOracleの統合が正しく動作することを確認

ステップ2: 短時間テスト（各パターン10k steps）
  for seed in 0; do
    python train.py task=pendulum-swingup seed=$seed steps=10000
    python train.py task=pendulum-swingup-randomized seed=$seed steps=10000
    python train.py task=pendulum-swingup use_oracle=true seed=$seed steps=10000
    python train.py task=pendulum-swingup-randomized use_oracle=true seed=$seed steps=10000
  done
  → 全パターンが正常に動作することを確認

ステップ3: 本格実験（各パターン500k steps × 3 seeds）
  for seed in 0 1 2; do
    python train.py task=pendulum-swingup seed=$seed steps=500000 &
    python train.py task=pendulum-swingup-randomized seed=$seed steps=500000 &
    python train.py task=pendulum-swingup use_oracle=true seed=$seed steps=500000 &
    python train.py task=pendulum-swingup-randomized use_oracle=true seed=$seed steps=500000 &
  done
```

### 期待される結果

| モデル | タスク | 期待性能 | 用途 |
|--------|--------|---------|------|
| Baseline | swingup | ~600 | 基本性能 |
| Baseline | swingup-randomized | ~500-550 | DR性能 |
| **Oracle** | swingup | **~800+** | 固定環境の上限 |
| **Oracle** | swingup-randomized | **~700-750** | DR環境の上限 |

**重要な比較:**
- **Oracle vs Baseline (同じタスク):** 物理情報の価値
- **swingup vs swingup-randomized (同じモデル):** DRによる難易度増加
- **Oracle (DR) の性能:** Model Cが目指すべき目標

---

## 📊 結果の保存先

```
tdmpc2/logs/
├── pendulum-swingup/
│   └── 0/                      # seed 0
│       ├── train.csv           # 学習ログ
│       ├── eval.csv            # 評価ログ
│       ├── config.yaml         # 使用した設定
│       └── model.pt            # チェックポイント（評価用）
│
└── pendulum-swingup-randomized/
    └── 0/
        ├── train.csv
        ├── eval.csv
        ├── config.yaml
        └── model.pt
```

**Oracle版の結果:**
- `use_oracle=true` を指定すると、ログディレクトリ名に `_oracle` が追加される
- 例: `logs/pendulum-swingup_oracle/0/`

---

## 📈 結果の可視化

### Pythonスクリプト例

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# データ読み込み
baseline = pd.read_csv('logs/pendulum-swingup/0/train.csv')
baseline_dr = pd.read_csv('logs/pendulum-swingup-randomized/0/train.csv')
oracle = pd.read_csv('logs/pendulum-swingup_oracle/0/train.csv')
oracle_dr = pd.read_csv('logs/pendulum-swingup-randomized_oracle/0/train.csv')

# プロット
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(baseline['step'], baseline['episode_reward'], 
        label='Baseline (fixed)', alpha=0.7, linewidth=2)
ax.plot(baseline_dr['step'], baseline_dr['episode_reward'], 
        label='Baseline (DR)', alpha=0.7, linewidth=2)
ax.plot(oracle['step'], oracle['episode_reward'], 
        label='Oracle (fixed)', alpha=0.7, linewidth=2)
ax.plot(oracle_dr['step'], oracle_dr['episode_reward'], 
        label='Oracle (DR)', alpha=0.7, linewidth=2, linestyle='--')

ax.set_xlabel('Training Steps', fontsize=14)
ax.set_ylabel('Episode Reward', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_title('Pendulum Swingup: Baseline vs Oracle', fontsize=16)

plt.tight_layout()
plt.savefig('pendulum_comparison.png', dpi=300)
print('Saved: pendulum_comparison.png')
```

---

## 🐛 トラブルシューティング

### エラー1: `ImportError: Oracle mode is enabled but...`

**原因:** Oracleコンポーネントのファイルが見つからない

**解決:**
```bash
# 必要なファイルを確認
ls tdmpc2/common/buffer_oracle.py
ls tdmpc2/common/world_model_oracle.py
ls tdmpc2/tdmpc2_oracle.py
ls tdmpc2/trainer/online_trainer_oracle.py
ls tdmpc2/envs/wrappers/physics_param.py
```

### エラー2: 物理パラメータが正しく取得できない

**原因:** 環境の構造が想定と異なる

**デバッグ:**
```bash
# 環境の構造を確認
python inspect_task.py pendulum swingup

# テストスクリプトを実行
python test_oracle_with_dr.py
```

### エラー3: DRで質量が変わらない

**原因:** タスク名が間違っている

**解決:**
```bash
# 正しい: pendulum-swingup-randomized
python train.py task=pendulum-swingup-randomized use_oracle=true

# 間違い: pendulum-swingup（DRなし）
python train.py task=pendulum-swingup use_oracle=true
```

### エラー4: CUDA out of memory

**原因:** GPUメモリ不足

**解決:**
```bash
# バッチサイズを減らす
python train.py task=pendulum-swingup use_oracle=true batch_size=128

# またはモデルサイズを小さくする
python train.py task=pendulum-swingup use_oracle=true model_size=1
```

---

## 💡 重要なポイント

### 1. Oracleの動作原理

```python
# エピソード開始時
obs = env.reset()
c_phys = env.current_c_phys  # ← DRで設定された真の質量を取得

# プランニング時（MPPI）
for t in range(horizon):
    reward = model.reward(z, action, task, c_phys)  # ← 真の質量を使用
    z = model.next(z, action, task, c_phys)         # ← 真の質量を使用

# 学習時
obs, action, reward, ..., c_phys = buffer.sample()  # ← バッファから取得
loss = agent.update(..., c_phys)                    # ← 真の質量を使用
```

**ポイント:**
- 推論時も学習時も、**同じ真の物理パラメータ**を使用
- DRで質量が変わっても、その時点の真の値を正しく取得

### 2. なぜDR + Oracleが重要？

| 設定 | 訓練環境 | 物理情報 | 意義 |
|------|---------|---------|------|
| Baseline (固定) | 質量=1.0 | なし | 基本性能 |
| Baseline (DR) | 質量変動 | なし | 汎化性能 |
| Oracle (固定) | 質量=1.0 | 完璧 | 固定環境の上限 |
| **Oracle (DR)** | 質量変動 | 完璧 | **変動環境の上限（Model Cの目標）** |

**DR + Oracleが最も重要な理由:**
- 実世界では物理パラメータは未知かつ変動する
- Model Cは「物理パラメータを推定して適応する」ことを目指す
- Oracle (DR) は「推定が完璧ならどこまで到達できるか」の上限を示す

---

## 📚 関連ドキュメント

- `README_ORACLE.md` - Oracleの詳細実装ガイド
- `docs/model_o_oracle_implementation.md` - 実装の詳細
- `test_oracle_quick.py` - 基本動作確認テスト
- `test_oracle_with_dr.py` - DR統合テスト

---

## ✅ チェックリスト

### 実験前
- [ ] `python test_oracle_with_dr.py` が成功
- [ ] DRで質量がランダム化されることを確認
- [ ] Oracleが真の質量を取得できることを確認

### 短時間テスト（10k steps）
- [ ] Baseline (固定) が正常に動作
- [ ] Baseline (DR) が正常に動作
- [ ] Oracle (固定) が正常に動作
- [ ] Oracle (DR) が正常に動作

### 本格実験（500k steps × 3 seeds）
- [ ] 全12実験（4パターン × 3 seeds）を実行
- [ ] チェックポイント（`.pt`）が保存されている
- [ ] ログファイル（`.csv`）が生成されている

### 結果分析
- [ ] 学習曲線をプロット
- [ ] 最終性能を比較
- [ ] Oracle vs Baseline の性能差を定量化

---

## 🎯 次のステップ

1. **動作確認（今すぐ）**
   ```bash
   python test_oracle_with_dr.py
   ```

2. **短時間テスト（1-2時間）**
   ```bash
   python tdmpc2/train.py task=pendulum-swingup-randomized use_oracle=true seed=0 steps=10000
   ```

3. **本格実験（1-2日）**
   ```bash
   # 複数シード×複数パターン
   for seed in 0 1 2; do
     python tdmpc2/train.py task=pendulum-swingup-randomized use_oracle=true seed=$seed steps=500000
   done
   ```

4. **Model Cの実装（次回）**
   - GRU推定器のオフライン検証
   - Model Cの統合
   - 最終評価

---

**実装完了日:** 2025-11-11  
**対応タスク:** Pendulum-Swingup (固定 & DR)  
**コマンドライン引数:** `use_oracle=true/false`

