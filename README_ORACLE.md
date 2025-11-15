# Model O (Oracle) - 完全ガイド

## 🎯 概要

**Model O (Oracle)** は、真の物理パラメータを常にプランナーに注入することで、**「物理推定が完璧な場合の理論的上限」**を検証するためのモデルです。

これは研究ロードマップにおける**ステップ2**に対応し、提案手法（Model C）の理論的ポテンシャルを明らかにします。

---

## 📦 実装ファイル

```
tdmpc2/
├── envs/wrappers/
│   └── physics_param.py              # 物理パラメータ取得Wrapper
├── common/
│   ├── buffer_oracle.py              # Oracle用Buffer
│   └── world_model_oracle.py         # 物理パラメータ条件付きWorldModel
├── trainer/
│   └── online_trainer_oracle.py      # Oracle用Trainer
├── tdmpc2_oracle.py                  # Oracle版TDMPC2
├── train_oracle.py                   # 学習スクリプト
└── config_oracle.yaml                # 設定ファイル

test_oracle_quick.py                   # クイックテスト
docs/model_o_oracle_implementation.md  # 詳細ドキュメント
```

---

## 🚀 クイックスタート

### 1. 動作確認（推奨）

```bash
# クイックテストで実装が正常に動作するか確認
python test_oracle_quick.py
```

**期待される出力:**
```
======================================================================
Model O (Oracle) Quick Test
======================================================================

[Test 1] Physics Parameter Wrapper
  ✓ Wrapper created
  ✓ Physics parameter extracted
  ✓ Test passed!

[Test 2] Oracle Buffer
  ✓ Buffer created
  ✓ Episode added to buffer
  ✓ Batch sampled
  ✓ Test passed!

...

======================================================================
✓ All tests passed!
======================================================================
```

### 2. 基本的な学習

```bash
cd tdmpc2
python train_oracle.py task=pendulum-swingup seed=0
```

### 3. 複数シードで実行

```bash
cd tdmpc2
for seed in 0 1 2; do
    python train_oracle.py task=pendulum-swingup seed=$seed
done
```

---

## 🔬 実験デザイン

### Model O の位置づけ

```
Model B (Baseline)
  ↓ 性能差 = 物理情報の価値
Model O (Oracle) ← 理論的上限
  ↓ 推定誤差による損失
Model C (提案手法)
```

### 比較実験の手順

#### ステップ1: ベースライン (Model B)
```bash
cd tdmpc2
python train.py task=pendulum-swingup seed=0 steps=500000
```

#### ステップ2: Oracle (Model O)
```bash
cd tdmpc2
python train_oracle.py task=pendulum-swingup seed=0 steps=500000
```

#### ステップ3: 結果の比較
```python
import pandas as pd
import matplotlib.pyplot as plt

# ベースラインの結果
baseline = pd.read_csv('logs/pendulum-swingup/0/train.csv')

# Oracleの結果
oracle = pd.read_csv('logs_oracle/pendulum-swingup/0/train.csv')

# 性能比較プロット
plt.plot(baseline['step'], baseline['episode_reward'], label='Model B (Baseline)')
plt.plot(oracle['step'], oracle['episode_reward'], label='Model O (Oracle)')
plt.xlabel('Training Steps')
plt.ylabel('Episode Reward')
plt.legend()
plt.savefig('baseline_vs_oracle.png')
```

---

## ⚙️ 設定オプション

### タスクの変更
```bash
# Ball-in-Cup
python train_oracle.py task=ball_in_cup-catch

# Hopper
python train_oracle.py task=hopper-stand

# Reacher
python train_oracle.py task=reacher-three_easy
```

### 物理パラメータの設定
```bash
# 正規化方法の変更
python train_oracle.py phys_param_normalization=minmax

# 特定の物理パラメータを指定
python train_oracle.py phys_param_indices="[-1]"

# 複数の物理パラメータ（Hopperなど）
python train_oracle.py task=hopper-stand c_phys_dim=3
```

### 学習設定
```bash
# 長い学習
python train_oracle.py steps=1000000

# 大きなモデル
python train_oracle.py model_size=19

# 評価頻度の変更
python train_oracle.py eval_freq=5000
```

---

## 📊 期待される結果

### 定量的指標

| モデル | Episode Reward | 成功率 |
|--------|---------------|--------|
| Model B (Baseline) | ~600 | 60% |
| **Model O (Oracle)** | **~800+** | **80%+** |

**重要な観察:**
- Model Oは**完璧な物理情報**を持つため、Model Bを大きく上回るはず
- この性能差が、Model C（提案手法）の**理論的上限**を示す

### 定性的観察

**Model O の特徴:**
1. **高速な学習:** 物理情報により探索が効率化
2. **安定した性能:** 物理法則の変動に頑健
3. **汎化性能:** 異なる質量でも適応可能

---

## 🐛 トラブルシューティング

### エラー1: `ModuleNotFoundError: No module named 'envs'`

**原因:** `tdmpc2/` ディレクトリ内で実行していない

**解決:**
```bash
cd tdmpc2
python train_oracle.py task=pendulum-swingup
```

### エラー2: `RuntimeError: CUDA out of memory`

**原因:** GPUメモリ不足

**解決:**
```bash
# バッチサイズを減らす
python train_oracle.py batch_size=128

# またはモデルサイズを小さくする
python train_oracle.py model_size=1
```

### エラー3: `KeyError: 'c_phys'`

**原因:** TensorDictに`c_phys`が含まれていない

**解決:** `to_td()`メソッドで`c_phys`が正しく追加されているか確認
```python
# trainer/online_trainer_oracle.py
td = TensorDict(
    obs=obs,
    action=action,
    reward=reward,
    terminated=terminated,
    c_phys=c_phys,  # ← 必須
    batch_size=(1,)
)
```

### エラー4: 物理パラメータが取得できない

**原因:** 環境の構造が想定と異なる

**デバッグ:**
```bash
# 環境の構造を確認
python inspect_task.py pendulum swingup

# 物理パラメータのインデックスを明示的に指定
python train_oracle.py phys_param_indices="[-1]"
```

---

## 📈 進捗の確認

### ログファイル

```
tdmpc2/logs_oracle/
└── pendulum-swingup/
    └── 0/
        ├── train.csv      # 学習ログ
        ├── eval.csv       # 評価ログ
        └── config.yaml    # 使用した設定
```

### リアルタイムモニタリング

```bash
# 学習中のログを監視
tail -f tdmpc2/logs_oracle/pendulum-swingup/0/train.csv

# または、TensorBoardを使用（オプション）
tensorboard --logdir=tdmpc2/logs_oracle
```

### 評価の実行

```bash
# 学習済みモデルの評価
python evaluate.py \
    task=pendulum-swingup \
    checkpoint=logs_oracle/pendulum-swingup/0/model.pt
```

---

## 📚 詳細ドキュメント

- **実装の詳細:** `docs/model_o_oracle_implementation.md`
- **研究ロードマップ:** プロンプト内で説明済み
- **ベースライン実験:** `docs/tdmpc2_baseline_plan.md`

---

## 🔬 次のステップ

### ステップ1: Model O の性能検証 ✅
- [x] 実装完了
- [ ] 動作確認（クイックテスト）
- [ ] 本格的な学習（500k steps）
- [ ] Model B との比較

### ステップ2: Model C の実装（次回）
1. **GRU推定器のオフライン検証**
   - Model Bのリプレイバッファからデータ収集
   - GRUのハイパーパラメータ探索
   - 推定精度の評価

2. **Model C の統合**
   - GRU推定器 + Oracle WorldModel
   - 2フェーズ分離アーキテクチャ
   - 勾配分離の実装

3. **最終評価**
   - Model B vs Model O vs Model C
   - 異なる物理パラメータでのテスト

---

## ✅ チェックリスト

### 実装前
- [ ] CUDA環境が利用可能
- [ ] DMControl環境がインストール済み
- [ ] Pendulumタスクが動作確認済み

### 動作確認
- [ ] `python test_oracle_quick.py` が成功
- [ ] 物理パラメータが正しく取得できる
- [ ] 学習ループが正常に回る

### 実験実行
- [ ] Model B (Baseline) の学習完了
- [ ] Model O (Oracle) の学習完了
- [ ] 性能比較のプロット作成

### 論文準備
- [ ] 性能差の定量化
- [ ] 理論的上限の考察
- [ ] Model C への示唆

---

## 💡 ヒント

### 高速化テクニック
```bash
# Compileを有効化（安定動作確認後）
python train_oracle.py compile=true

# 小さいモデルで高速検証
python train_oracle.py model_size=1 steps=100000
```

### デバッグモード
```python
# tdmpc2_oracle.py に追加
def _plan(self, obs, c_phys, ...):
    print(f"DEBUG: c_phys = {c_phys}")
    print(f"DEBUG: c_phys shape = {c_phys.shape}")
    ...
```

### 可視化
```bash
# ビデオ保存を有効化
python train_oracle.py save_video=true

# 結果は logs_oracle/pendulum-swingup/0/videos/ に保存
```

---

## 📞 サポート

問題が発生した場合:
1. `test_oracle_quick.py` を実行して基本動作を確認
2. `docs/model_o_oracle_implementation.md` で詳細を確認
3. エラーメッセージとトレースバックを保存

---

**実装完了日:** 2025-11-11  
**対象タスク:** Pendulum-Swingup (他のタスクにも対応)  
**次のステップ:** 動作確認 → 本格実験 → Model C の実装

---

**Good luck with your experiments! 🚀**

