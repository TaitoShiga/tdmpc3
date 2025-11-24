# 統計解析パイプライン

4種類のモデル（baseline, dr, Model C, Oracle）を評価し、統計解析を行うためのスクリプト群。

## ディレクトリ構成

```
evaluate/
├── README.md                    # このファイル
├── evaluate_all_models.py       # 評価スクリプト
└── analyze_results.py           # 統計解析スクリプト
```

## 使用方法

### 1. 全モデルの評価

4種類のモデルを複数のseedと物理パラメータで評価し、`results.csv`を生成：

```bash
python evaluate/evaluate_all_models.py \
    --seeds 0 1 2 3 4 \
    --episodes 30 \
    --test-params 0.5 1.0 1.5 2.0 2.5 \
    --output results.csv
```

**パラメータ:**
- `--seeds`: 評価するseedのリスト（デフォルト: 0 1 2 3 4）
- `--episodes`: 各(model, seed, param)あたりの評価エピソード数（デフォルト: 30）
- `--test-params`: テストする物理パラメータ倍率（デフォルト: 0.5 1.0 1.5 2.0 2.5）
- `--output`: 出力CSVファイル（デフォルト: results.csv）
- `--logs-dir`: ログディレクトリ（デフォルト: logs_remote）
- `--task`: タスク名（デフォルト: pendulum-swingup）

**出力形式（results.csv）:**
| model | seed | param | episode | return | length | success |
|-------|------|-------|---------|--------|--------|---------|
| baseline | 0 | 1.0 | 0 | 850.2 | 1000 | 1.0 |
| baseline | 0 | 1.0 | 1 | 845.7 | 1000 | 1.0 |
| ... | ... | ... | ... | ... | ... | ... |

### 2. 統計解析

`results.csv`を読み込み、IQMによる統計解析と可視化を実行：

```bash
python evaluate/analyze_results.py
```

**実行内容:**
1. IQM（Interquartile Mean）の計算
   - 各(model, seed, param)について、30エピソードのreturnから1つのIQMスコアを算出
2. ブートストラップによる95%信頼区間の推定
   - 10,000回のブートストラップでseed間の変動を推定
3. モデル間のペアワイズ比較
   - 6ペア（baseline vs dr, baseline vs c, ...）について差分を計算
   - 統計的有意性を判定（CIが0をまたぐかどうか）
4. 可視化
   - `fig_per_param.png`: param別の性能比較（折れ線グラフ）
   - `fig_overall.png`: 全体性能比較（棒グラフ）

## 評価対象モデル

| モデル名 | 説明 | チェックポイント |
|---------|------|-----------------|
| **baseline** | 標準環境（mass=1.0）で学習 | `logs_remote/pendulum-swingup/{seed}/baseline/models/final.pt` |
| **dr** | Domain Randomizationで学習 | `logs_remote/pendulum-swingup-randomized/{seed}/dr/models/final.pt` |
| **c** | Model C（GRU推定器統合版） | `logs_remote/pendulum-swingup-randomized/{seed}/modelc/models/final.pt` |
| **o** | Oracle（真の物理パラメータ利用） | `logs_remote/pendulum-swingup-randomized/{seed}/oracle/models/final.pt` |

## テストパラメータ

pendulum-swingupタスクでは、質量倍率を変更：

| 倍率 | タスク名 | 説明 |
|------|---------|------|
| 0.5 | pendulum-swingup-mass05 | 質量0.5倍 |
| 1.0 | pendulum-swingup | デフォルト（訓練環境） |
| 1.5 | pendulum-swingup-mass15 | 質量1.5倍 |
| 2.0 | pendulum-swingup-mass2 | 質量2倍 |
| 2.5 | pendulum-swingup-mass25 | 質量2.5倍 |

## 統計指標の詳細

### IQM (Interquartile Mean)

上位25%と下位25%を除いた中央50%の平均値。外れ値に頑健な評価指標。

```python
def compute_iqm(values):
    sorted_values = np.sort(values)
    q25 = np.percentile(sorted_values, 25)
    q75 = np.percentile(sorted_values, 75)
    iqr_values = sorted_values[(sorted_values >= q25) & (sorted_values <= q75)]
    return np.mean(iqr_values)
```

### ブートストラップ95% CI

seed間の変動を考慮した信頼区間を推定。10,000回のリサンプリングで分布を構築。

## 軽量テスト

評価スクリプトが正しく動作するか確認するための軽量版：

```bash
# 1 seed × 1 param × 5 episodes でテスト
python evaluate/evaluate_all_models.py \
    --seeds 0 \
    --episodes 5 \
    --test-params 1.0 \
    --output results_test.csv
```

## トラブルシューティング

### チェックポイントが見つからない

```
Error: Checkpoint not found: logs_remote/...
```

→ `--logs-dir` パラメータで正しいディレクトリを指定してください。

### CUDA out of memory

→ `--episodes` を減らすか、1つずつ評価するようにスクリプトを修正してください。

### 環境が見つからない

```
Error: Could not create task 'pendulum-swingup-mass05'
```

→ `tdmpc2/envs/dmcontrol.py` に該当タスクの定義が必要です。

## 実装の詳細

- **評価スクリプト**: logs_remoteディレクトリから各モデルのチェックポイントをロードし、複数のparam値で評価
- **統計解析スクリプト**: IQMとブートストラップを使用した頑健な統計解析を実装
- **再現性**: `np.random.seed(0)`で乱数を固定（ブートストラップの再現性）

## 参考文献

- IQM: Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice", NeurIPS 2021
- Bootstrap CI: Efron & Tibshirani, "An Introduction to the Bootstrap", 1993

