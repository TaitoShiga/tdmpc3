# 評価・解析パイプライン使用例

## クイックスタート

### 1. 動作確認（クイックテスト）

まず、1つのモデルで動作確認：

```bash
python evaluate/quick_test.py
```

出力例：
```
======================================================================
クイックテスト: baseline モデル評価
======================================================================
チェックポイント: logs_remote/pendulum-swingup/0/baseline/models/final.pt
タスク: pendulum-swingup

✓ エージェントロード完了

Episode 0: return=850.23, length=1000
Episode 1: return=845.67, length=1000
Episode 2: return=852.11, length=1000
Episode 3: return=847.89, length=1000
Episode 4: return=849.45, length=1000

======================================================================
結果サマリー
======================================================================
平均リターン: 849.07 ± 2.34
最小/最大: 845.67 / 852.11

✓ テスト成功！
```

### 2. 軽量評価（1 seed × 1 param）

次に、4モデルで軽量評価：

```bash
python evaluate/evaluate_all_models.py \
    --seeds 0 \
    --episodes 5 \
    --test-params 1.0 \
    --output results_test.csv
```

**実行時間**: 約5-10分

**出力**: `results_test.csv`
```csv
model,seed,param,episode,return,length,success
baseline,0,1.0,0,850.2,1000,1.0
baseline,0,1.0,1,845.7,1000,1.0
...
```

### 3. 完全評価（5 seeds × 5 params × 30 episodes）

本番の評価：

```bash
python evaluate/evaluate_all_models.py \
    --seeds 0 1 2 3 4 \
    --episodes 30 \
    --test-params 0.5 1.0 1.5 2.0 2.5 \
    --output results.csv
```

**実行時間**: 約2-4時間（GPU使用時）

**出力**: `results.csv`（約18,000行）
- 4 モデル × 5 seeds × 5 params × 30 episodes = 3,000行
- 実際には一部のモデル・paramの組み合わせでチェックポイントが存在しない場合もある

### 4. 統計解析

`results.csv` を解析：

```bash
python evaluate/analyze_results.py
```

**実行時間**: 約30秒（ブートストラップ10,000回）

**出力**:
1. コンソール出力: 統計サマリー、ペアワイズ比較
2. `fig_per_param.png`: param別の性能比較
3. `fig_overall.png`: 全体性能比較

出力例：
```
======================================================================
解析完了
======================================================================
✓ param ごとの結果: 20 エントリ
✓ overall 結果: 4 エントリ
✓ ペアワイズ比較: 30 ペア

生成されたファイル:
  - fig_per_param.png
  - fig_overall.png

統計的に有意な差分:
  o vs baseline (param=overall): diff=+45.23 [38.12, 52.34]
  c vs baseline (param=1.5): diff=+12.45 [5.67, 19.23]
  dr vs baseline (param=2.0): diff=+8.90 [2.34, 15.46]
```

## 結果の解釈

### IQM (Interquartile Mean)

各(model, seed, param)について、30エピソードのreturnから上位25%と下位25%を除いた中央50%の平均値を計算。

**利点**:
- 外れ値に頑健
- 平均よりも安定した指標

### ブートストラップ95% CI

seed間の変動を考慮した信頼区間。

**解釈**:
- CIが狭い → 安定した性能
- CIが広い → seed間のばらつきが大きい
- CI同士が重ならない → 統計的に有意な差

### ペアワイズ比較

6ペアの差分を計算：
- `baseline vs dr`: DRの効果
- `baseline vs c`: Model Cの効果
- `baseline vs o`: Oracleとの差（推定の限界）
- `dr vs c`: Model C vs DRの比較
- `dr vs o`, `c vs o`: Oracleとの差

**significant = True**の場合、95%信頼区間が0をまたがない（統計的に有意）。

## トラブルシューティング

### エラー: Checkpoint not found

**原因**: 指定されたチェックポイントが存在しない

**対処法**:
1. `--logs-dir` パラメータを確認
2. チェックポイントのディレクトリ構造を確認：
   ```bash
   ls logs_remote/pendulum-swingup/0/baseline/models/
   ls logs_remote/pendulum-swingup-randomized/0/dr/models/
   ```

### エラー: CUDA out of memory

**原因**: GPUメモリ不足

**対処法**:
1. `--episodes` を減らす（30 → 10）
2. 1つずつ評価する：
   ```bash
   python evaluate/evaluate_all_models.py --seeds 0 --test-params 1.0
   python evaluate/evaluate_all_models.py --seeds 1 --test-params 1.0
   # ... 結果を結合
   ```

### エラー: Unknown task

**原因**: タスク定義が存在しない

**対処法**:
`tdmpc2/envs/tasks/pendulum.py` に該当タスクの定義を追加：
```python
@pendulum.SUITE.add('custom')
def swingup_mass05(time_limit=_DEFAULT_TIME_LIMIT, ...):
    physics = pendulum.Physics.from_xml_string(*get_model_and_assets(mass=0.5))
    ...
```

### 警告: IQM計算でNaN

**原因**: エピソード数が少なすぎる（< 4）

**対処法**: `--episodes` を増やす（最低10以上推奨）

## カスタマイズ

### 異なるタスクで評価

```python
# evaluate/evaluate_all_models.py の get_task_for_param() を修正
def get_task_for_param(base_task: str, param_multiplier: float) -> str:
    if base_task == "hopper-stand":
        # hopper用のタスク名マッピングを追加
        ...
```

### 異なる物理パラメータ

```python
# evaluate/evaluate_all_models.py の load_agent() を修正
# PhysicsParamWrapperのparam_typeを変更
env = PhysicsParamWrapper(env, param_type='friction', ...)
```

### ブートストラップ回数の変更

```python
# evaluate/analyze_results.py の BOOTSTRAP_ITERATIONS を変更
BOOTSTRAP_ITERATIONS = 5000  # デフォルトは10000
```

## 論文・発表用の図表

生成された `fig_per_param.png` と `fig_overall.png` はそのまま論文に使用可能（150 DPI）。

**さらに高品質な図が必要な場合**:
```python
# analyze_results.py の plot_per_param() と plot_overall() で
plt.savefig(output_path, dpi=300)  # 150 → 300
```

## 参考

- **IQM**: Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice", NeurIPS 2021
- **Bootstrap**: Efron & Tibshirani, "An Introduction to the Bootstrap", 1993

