# Model C (GRU推定器統合版) - 完全ガイド

## 🎯 概要

**Model C** は、GRU推定器を使って物理パラメータを推定し、それをプランナーに注入する提案手法です。

**核心的特徴:** 2フェーズ分離 + 勾配分離
- **フェーズ1:** GRU推定器が履歴から物理パラメータを推定（L_auxで更新）
- **フェーズ2:** プランナーが推定されたパラメータを使用（L_TD-MPC2で更新）
- **勾配分離:** `.detach()` で2つの学習を分離

---

## 📦 実装ファイル

```
tdmpc2/
├── common/
│   ├── physics_estimator.py         # GRU/MLP推定器
│   ├── world_model_model_c.py       # Model C用WorldModel
│   └── buffer_model_c.py            # 履歴保存Buffer
├── trainer/
│   └── online_trainer_model_c.py    # Model C用Trainer
├── tdmpc2_model_c.py                # Model Cエージェント（勾配分離実装）
├── train_gru_offline.py             # GRUオフライン学習
└── config_gru_offline.yaml          # GRU学習設定
```

---

## 🚀 使用方法

### ステップ1: GRUをオフライン学習（推奨）

```bash
cd tdmpc2

# DR環境からデータを収集してGRUを学習
python train_gru_offline.py \
    task=pendulum-swingup-randomized \
    num_episodes=1000 \
    gru_epochs=100 \
    context_length=50 \
    gru_hidden_dim=256
```

**出力:**
- `logs_gru/pendulum-swingup-randomized/0/best_gru.pt`
- 学習曲線と予測精度のプロット

### ステップ2: Model Cで統合学習

#### オプションA: GRUをゼロから学習
```bash
python train.py \
    task=pendulum-swingup-randomized \
    use_model_c=true \
    seed=0 \
    steps=500000
```

#### オプションB: 事前学習済みGRUをロード（推奨）
```bash
python train.py \
    task=pendulum-swingup-randomized \
    use_model_c=true \
    gru_pretrained=logs_gru/pendulum-swingup-randomized/0/best_gru.pt \
    seed=0 \
    steps=500000
```

---

## 🔬 3モデルの比較実験

### 完全な実験セット

```bash
cd tdmpc2

# Model B (Baseline - DR)
python train.py task=pendulum-swingup-randomized seed=0 steps=500000

# Model O (Oracle - 理論的上限)
python train.py task=pendulum-swingup-randomized use_oracle=true seed=0 steps=500000

# Model C (提案手法)
# ステップ1: GRU学習
python train_gru_offline.py task=pendulum-swingup-randomized num_episodes=1000

# ステップ2: 統合学習
python train.py \
    task=pendulum-swingup-randomized \
    use_model_c=true \
    gru_pretrained=logs_gru/pendulum-swingup-randomized/0/best_gru.pt \
    seed=0 \
    steps=500000
```

---

## 📊 期待される結果

### Pendulum-Swingup-Randomized

| モデル | Episode Reward | 意義 |
|--------|---------------|------|
| Model B (Baseline) | ~500-550 | DRのみ |
| **Model C (提案手法)** | **~650-700** | **GRU推定 + 適応** |
| Model O (Oracle) | ~700-750 | 理論的上限 |

**重要な比較:**
- **Model C - Model B**: GRU推定器の効果
- **Model O - Model C**: 推定誤差による損失
- **目標:** Model C を Model B より大きく改善し、Model O に近づける

---

## 🔑 勾配分離の実装

Model Cの核心的特徴：

```python
# tdmpc2_model_c.py の _update() メソッド

# ========================================
# フェーズ1: GRU推定器の更新（L_aux）
# ========================================
loss_aux, info_aux = model.compute_physics_estimation_loss(
    obs_seq, action_seq, c_phys_true
)

# GRU推定器のみ更新
gru_optim.zero_grad()
loss_aux.backward()  # ← GRUの勾配のみ
gru_optim.step()

# ========================================
# フェーズ2: プランナーの更新（L_TD-MPC2）
# ========================================

# 🔑 重要: detach()で勾配を切る
c_phys_pred = model.estimate_physics(obs_seq, action_seq)
c_phys = c_phys_pred.detach()  # ← 勾配を分離！

# プランナーで使用
z = model.next(z, action, task, c_phys)  # detach済みのc_physを使用
reward = model.reward(z, action, task, c_phys)
Q = model.Q(z, action, task, c_phys)

# プランナーのみ更新
total_loss.backward()  # ← プランナーの勾配のみ
planner_optim.step()
```

---

## ⚙️ ハイパーパラメータ

### GRU推定器
```yaml
context_length: 50          # 履歴長（何ステップ見るか）
gru_hidden_dim: 256         # GRU隠れ層次元
gru_num_layers: 2           # GRU層数
gru_dropout: 0.1            # ドロップアウト率
gru_lr: 3e-4                # 学習率
```

### オフライン学習
```yaml
num_episodes: 1000          # 収集するエピソード数
gru_epochs: 100             # 学習エポック数
gru_batch_size: 128         # バッチサイズ
```

### チューニングのヒント
- **context_length**: 短すぎると情報不足、長すぎると学習困難
  - Pendulum: 30-50
  - Ball-in-Cup: 50-100
- **gru_hidden_dim**: 大きいほど表現力が高いが、学習が遅い
  - 小: 128, 中: 256, 大: 512

---

## 📈 ログとモニタリング

### 学習ログ

```
tdmpc2/logs/
├── pendulum-swingup-randomized_model_c/
│   └── 0/
│       ├── train.csv              # 学習ログ
│       ├── eval.csv               # 評価ログ
│       └── model.pt               # チェックポイント
│
└── logs_gru/
    └── pendulum-swingup-randomized/
        └── 0/
            ├── best_gru.pt                    # GRUモデル
            ├── gru_training_curve.png         # 学習曲線
            └── gru_prediction_vs_truth.png    # 予測精度
```

### 重要な指標

**GRUオフライン学習:**
- `val_mae`: 検証セットでのMAE（小さいほど良い）
- `val_loss`: 検証セットでのMSE損失

**Model C統合学習:**
- `gru_loss_aux`: GRU推定損失（L_aux）
- `gru_mae`: 物理パラメータ推定のMAE
- `total_loss`: プランナーの制御損失（L_TD-MPC2）
- `episode_reward`: エピソード報酬

---

## 🐛 トラブルシューティング

### エラー1: GRUの推定精度が低い

**症状:** `val_mae` が大きい（例: > 0.5）

**原因と解決策:**
1. **データ不足**
   ```bash
   # より多くのエピソードを収集
   python train_gru_offline.py num_episodes=2000
   ```

2. **履歴長が不適切**
   ```bash
   # context_lengthを調整
   python train_gru_offline.py context_length=100
   ```

3. **正規化の問題**
   - PhysicsParamWrapperの正規化設定を確認

### エラー2: Model Cの学習が不安定

**症状:** 損失が発散、または性能がModel Bより悪い

**原因と解決策:**
1. **勾配分離の実装を確認**
   - `c_phys.detach()` が正しく呼ばれているか

2. **GRUの事前学習を使用**
   ```bash
   # ゼロから学習せず、事前学習済みGRUをロード
   python train.py use_model_c=true gru_pretrained=logs_gru/.../best_gru.pt
   ```

3. **学習率を下げる**
   ```bash
   python train.py use_model_c=true gru_lr=1e-4
   ```

### エラー3: CUDA out of memory

**原因:** 履歴データがメモリを圧迫

**解決:**
```bash
# バッチサイズを減らす
python train.py use_model_c=true batch_size=128

# または context_length を短くする
python train.py use_model_c=true context_length=30
```

---

## 💡 実装のポイント

### 1. 履歴管理

```python
# OnlineTrainerModelC
class OnlineTrainerModelC:
    def __init__(self):
        self._obs_history = []
        self._action_history = []
    
    def _get_history_window(self, t):
        # 最新のcontext_length分を取得
        obs_window = self._obs_history[-self.context_length:]
        # 不十分な場合はゼロパディング
```

### 2. Buffer

```python
# ModelCBuffer
# 各ステップに履歴を保存
td = TensorDict(
    obs=obs,
    action=action,
    reward=reward,
    c_phys=c_phys,  # 真の値（GRU学習用）
    obs_history=obs_window,  # GRU入力用
    action_history=action_window,
    ...
)
```

### 3. 2つのOptimizer

```python
# TDMPC2ModelC
self.gru_optim = Adam(model._physics_estimator.parameters())
self.optim = Adam([
    model._encoder.parameters(),
    model._dynamics.parameters(),
    model._reward.parameters(),
    model._Qs.parameters(),
])
```

---

## 📚 詳細ドキュメント

- **実装の詳細:** `docs/model_o_oracle_implementation.md`
- **研究ロードマップ:** プロンプト内で説明済み
- **Oracle実装:** `README_ORACLE.md`

---

## ✅ チェックリスト

### GRUオフライン学習
- [ ] データ収集が完了（1000エピソード）
- [ ] `val_mae < 0.2` を達成
- [ ] `best_gru.pt` が保存されている

### Model C統合学習
- [ ] 事前学習済みGRUをロード
- [ ] 学習が安定している（損失が発散しない）
- [ ] Model B より性能が向上

### 最終評価
- [ ] Model B, O, C の3つを比較
- [ ] Model C が Model B を上回る
- [ ] Model O との差（推定誤差）を定量化

---

## 🎓 論文用の重要な結果

### 定量的評価
1. **性能比較:** Model B < Model C < Model O
2. **GRU推定精度:** MAE, 予測vs真値のプロット
3. **学習曲線:** 3モデルの学習速度比較

### 定性的評価
1. **勾配分離の効果:** L_auxとL_TD-MPC2の独立学習
2. **2フェーズ分離:** 推定と制御の責任分離
3. **汎化性能:** 異なる質量での適応能力

---

**実装完了日:** 2025-11-11  
**対象タスク:** Pendulum-Swingup-Randomized  
**次のステップ:** GRUオフライン学習 → Model C統合学習 → 性能比較

---

**Good luck with your research! 🚀**

