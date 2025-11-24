# 統計解析パイプライン

4種類のモデル（baseline, DR, Model C, Oracle）を評価し、IQMとブートストラップによる統計解析を行うパイプライン。

## ディレクトリ構成

```
evaluate/
├── README.md                    # 詳細なドキュメント
├── USAGE_EXAMPLE.md             # 使用例とトラブルシューティング
├── evaluate_all_models.py       # 評価スクリプト（results.csv生成）
├── analyze_results.py           # 統計解析スクリプト（IQM + Bootstrap CI）
└── quick_test.py                # 動作確認用クイックテスト
```

## クイックスタート

### 1. 動作確認

```bash
python evaluate/quick_test.py
```

### 2. 軽量評価（テスト）

```bash
python evaluate/evaluate_all_models.py --seeds 0 --episodes 5 --test-params 1.0 --output results_test.csv
```

### 3. 完全評価（本番）

```bash
# 評価実行（約2-4時間）
python evaluate/evaluate_all_models.py \
    --seeds 0 1 2 3 4 \
    --episodes 30 \
    --test-params 0.5 1.0 1.5 2.0 2.5 \
    --output results.csv

# 統計解析（約30秒）
python evaluate/analyze_results.py
```

### 4. 結果確認

- `results.csv`: 生データ（約18,000行）
- `fig_per_param.png`: param別の性能比較グラフ
- `fig_overall.png`: 全体性能比較グラフ

## 評価対象モデル

| モデル | 説明 | チェックポイント |
|-------|------|-----------------|
| **baseline** | 標準環境（mass=1.0）で学習 | `logs_remote/pendulum-swingup/{seed}/baseline/models/final.pt` |
| **dr** | Domain Randomizationで学習 | `logs_remote/pendulum-swingup-randomized/{seed}/dr/models/final.pt` |
| **c** | Model C（GRU推定器統合版） | `logs_remote/pendulum-swingup-randomized/{seed}/modelc/models/final.pt` |
| **o** | Oracle（真の物理パラメータ利用） | `logs_remote/pendulum-swingup-randomized/{seed}/oracle/models/final.pt` |

## 統計指標

### IQM (Interquartile Mean)

各(model, seed, param)について30エピソードのreturnから上位25%と下位25%を除いた中央50%の平均値。

### ブートストラップ95% CI

10,000回のリサンプリングでseed間の変動を考慮した信頼区間を推定。

### ペアワイズ比較

6ペア（baseline vs dr, baseline vs c, ...）について差分と統計的有意性を計算。

## 実装の詳細

### evaluate_all_models.py

**入力**: logs_remoteディレクトリのチェックポイント  
**出力**: results.csv

**カラム構成**:
- `model`: "baseline", "dr", "c", "o"
- `seed`: 0, 1, 2, 3, 4
- `param`: 0.5, 1.0, 1.5, 2.0, 2.5（質量倍率）
- `episode`: 0〜29
- `return`: エピソードリターン
- `length`: エピソード長
- `success`: 成功フラグ

### analyze_results.py

**入力**: results.csv  
**出力**: 統計解析結果 + グラフ

**処理の流れ**:
1. データ読み込み・前処理
2. IQM計算（各(model, seed, param)で1つのスコア）
3. ブートストラップによる95% CI推定
4. ペアワイズ比較
5. 可視化

## 統計解析の仕様

### 前提：実験データの形式

- `results.csv`という1つのCSVファイル
- 各行は「1エピソードの評価結果」
- カラム: model, seed, param, episode, return, length, success

### IQMの計算

- 解析の基本単位は「seed」
- 各(model, seed, param)について、エピソードのreturnからIQMを計算

### ブートストラップCI

- 各モデルの平均性能について95%信頼区間を推定
- ブートストラップ回数: 10,000回
- リサンプリング単位: seed

### モデル間のペアワイズ差分

- 順序関係を仮定せず、6ペアすべてを比較
- 差分は同じseedに対応するペアの差
- ブートストラップでCIを推定し、統計的有意性を判定

### 可視化

1. **param別の性能比較**: 折れ線グラフ（x軸=param, y軸=IQM, 各モデルを色分け）
2. **overall性能比較**: 棒グラフ（x軸=model, y軸=param平均後のIQM）

## 詳細なドキュメント

- [evaluate/README.md](evaluate/README.md) - 詳細な仕様と実装説明
- [evaluate/USAGE_EXAMPLE.md](evaluate/USAGE_EXAMPLE.md) - 使用例とトラブルシューティング

## 参考文献

- **IQM**: Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice", NeurIPS 2021
- **Bootstrap CI**: Efron & Tibshirani, "An Introduction to the Bootstrap", 1993

