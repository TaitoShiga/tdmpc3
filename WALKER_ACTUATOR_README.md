# Walker Actuator Randomization 実験ガイド

## 概要

Walker環境のアクチュエーターgear強度をランダム化し、4つのモデル（Baseline, DR, Model C, Oracle）のロバスト性と適応性を評価する実験。

## 学習設定

### 利用可能なチェックポイント

**重要**: 現在利用可能なseedが限定されています：
- **Baseline**: seed0のみ
- **DR**: seed3のみ  
- **Model C**: seed0のみ
- **Oracle**: seed0のみ

評価スクリプトは自動的にこれらのseedのみを使用します。

### アクチュエーター強度の範囲

**学習時（Domain Randomization）:**
- 範囲: **0.4x ~ 1.4x** (uniform random)
- 定義: `tdmpc2/envs/tasks/walker.py` の `_ACTUATOR_SCALE_RANGE`
- エピソードごとに全アクチュエーターのgearを同じスケールで調整

**デフォルトgear値 (walker.xml):**
```
- right_hip: 100
- right_knee: 50
- right_ankle: 20
- left_hip: 100
- left_knee: 50
- left_ankle: 20
```

### 学習タスク

| モデル | タスク | アクチュエーター | 特徴 |
|--------|--------|------------------|------|
| **Baseline** | `walker-walk` | 固定 (1.0x) | 標準環境で学習 |
| **DR** | `walker-walk_actuator_randomized` | ランダム (0.4-1.4x) | Domain Randomization |
| **Model C** | `walker-walk_actuator_randomized` | ランダム (0.4-1.4x) | In-Context Learning |
| **Oracle** | `walker-walk_actuator_randomized` | ランダム (0.4-1.4x) | 真のパラメータ使用 |

## 評価設定

### 評価範囲

**固定アクチュエーター強度での評価:**

**In-Distribution (学習範囲内):**
- 範囲: **0.4x, 0.5x, 0.6x, 0.7x, 0.8x, 0.9x, 1.0x, 1.1x, 1.2x, 1.3x, 1.4x** (11点)
- タスク名: `walker-walk_actuator_{04-14}x`

**Out-of-Distribution (学習範囲外):**
- 軽すぎ: **0.2x, 0.3x** (2点)
- 重すぎ: **1.5x, 1.6x, 1.7x** (3点)
- タスク名: `walker-walk_actuator_{02,03,15,16,17}x`

各設定で30エピソード実行。

### 評価区間の関係（Pendulum同様）

```
         OOD          学習範囲 (DR)             OOD
    <─────────> <─────────────────────────> <─────────>
    0.2  0.3    0.4 ............... 1.4     1.5  1.6  1.7
    ├────┼──────┼──────────────────┼────────┼────┼────┤
    軽すぎ      In-Distribution      重すぎ（強すぎ）
```

- **In-distribution (11点)**: 学習範囲内、補間性能を評価
- **OOD (5点)**: 学習範囲外、汎化性能を評価

## 実行手順

### 1. 学習（Slurmクラスタ）

```bash
# 全モデル・全seed（20ジョブ）を一括投入
cd slurm_scripts
bash train_walker_actuator_models.sh

# または個別に
sbatch job_walker_actuator_dr_seed0.sh
sbatch job_walker_actuator_model_c_seed0.sh
sbatch job_walker_actuator_oracle_seed0.sh
```

**学習パラメータ:**
- Steps: 100,000
- Eval frequency: 500 steps
- Seeds: 0-4

### 2. 評価実行

```bash
# デフォルト: In-Dist + OOD全て（16点 × 利用可能seed）
sbatch slurm_scripts/job_walker_actuator_eval.sh

# In-Distributionのみ評価（11点）
IN_DIST_ONLY=true sbatch slurm_scripts/job_walker_actuator_eval.sh

# 出力先指定
OUTPUT=results_walker_actuator_full.csv sbatch slurm_scripts/job_walker_actuator_eval.sh
```

**評価実行時間:** 
- In-Dist + OOD (16点): 約4-5時間（利用可能seedのみ）
- In-Distのみ (11点): 約3時間

**注意**: 利用可能なseed（Baseline:0, DR:3, C:0, O:0）のみ自動評価されます。

### 3. 結果確認

```bash
# 評価が完了したら結果を確認
wc -l results_walker_actuator.csv
# 期待: 16 scales × 30 episodes × 4 models = 1920 + 1 header = 1921 lines

# 各モデルのデータ数を確認
cut -d, -f1 results_walker_actuator.csv | tail -n +2 | sort | uniq -c
# 期待: baseline=480, dr=480, c=480, o=480
```

### 4. 可視化・分析

```bash
# 評価プロットの生成（Slurmクラスタ）
RESULTS=results_walker_actuator.csv sbatch slurm_scripts/job_walker_actuator_plots.sh

# またはローカルで実行（OOD範囲を網掛け表示）
python evaluate/analyze_results.py \
    --input results_walker_actuator.csv \
    --output-dir . \
    --output-prefix walker_actuator \
    --param-label "Actuator scale (x)" \
    --train-min 0.4 \
    --train-max 1.4

# 学習曲線のプロット
python plot_learning_curves.py \
    --task walker_actuator \
    --output eval_curves_walker_actuator.png \
    --smooth gaussian --sigma 2.0
```

**生成される図（Pendulum実験と同様）:**
1. `walker_actuator_overall.png` - 全体性能比較（IQM + 95% CI）
2. `walker_actuator_per_param.png` - パラメータ別性能曲線（OOD範囲を網掛け）
3. `walker_actuator_degradation.png` - Baselineからの性能劣化分析
4. `walker_actuator_ood.png` - OOD vs In-Dist比較
5. `walker_actuator_heatmap.png` - モデル×スケールのヒートマップ
6. `eval_curves_walker_actuator.png` - 学習曲線

## 出力ファイル

### 評価結果

**CSV形式** (`results_walker_actuator.csv`):
```csv
model,seed,param,episode,return,length,success
baseline,0,0.4,0,120.5,1000,0.0
baseline,0,0.4,1,115.3,1000,0.0
...
```

### 生成される図

1. **walker_actuator_overall.png**: 全モデルの性能比較（IQM）
2. **walker_actuator_per_param.png**: パラメータごとの詳細比較
3. **walker_actuator_degradation.png**: Baselineからの性能劣化
4. **walker_actuator_ood.png**: OOD評価（該当なし）
5. **walker_actuator_heatmap.png**: seed×paramのヒートマップ
6. **eval_curves_walker_actuator.png**: 学習曲線

### 学習ログ

```
logs/
└── walker-walk_actuator_randomized/
    ├── 0/walker_actuator_dr/
    │   ├── eval.csv          # 学習中の評価
    │   └── models/final.pt   # 学習済みモデル
    ├── 0/walker_actuator_model_c/
    └── 0/walker_actuator_oracle/
```

### Artifacts（学習曲線用）

```
artifacts/
├── walker_dr/seed0.csv
├── walker_c/seed0.csv
└── walker_oracle/seed0.csv
```

## 予想される結果

### 性能順位（予想）

1. **Oracle** > **Model C** ≥ **DR** > **Baseline**
2. Baselineはアクチュエーター強度が変わると大幅に性能低下
3. Model CはOracleに近い性能を発揮（In-Context Learning効果）
4. DRは全範囲で安定するが、最高性能はModel C/Oracleに劣る

### 評価メトリクス

- **IQM (Interquartile Mean)**: 上位25%・下位25%を除いた平均
- **95% Confidence Interval**: ブートストラップ法
- **Degradation**: Baseline比での性能低下率

## トラブルシューティング

### conda/python not found

スクリプト内で以下を試す:
```bash
# Option 1
source ~/.bashrc
conda activate tdmpc2

# Option 2
eval "$(conda shell.bash hook)"
conda activate tdmpc2

# Option 3 (module環境の場合)
module load anaconda3
conda activate tdmpc2
```

### チェックポイントが見つからない

```bash
# 期待されるパス
logs/walker-walk_actuator_randomized/{seed}/walker_actuator_{model}/models/final.pt

# 存在確認
ls -la logs/walker-walk_actuator_randomized/0/walker_actuator_*/models/
```

### メモリ不足

評価スクリプトのSBATCH設定を変更:
```bash
#SBATCH --mem=64G  # 32G → 64G
```

## 参考実験: Pendulum

同様のドメインランダム化実験をPendulumで実施済み:
- タスク: `pendulum-swingup-randomized`
- パラメータ: 質量 (0.5x ~ 2.5x)
- 学習ステップ: 100,000
- 評価: 5段階 + OOD 3段階

Walkerはより複雑な環境で、アクチュエーター強度という新しい摂動軸を評価。

---

**作成日**: 2025年1月4日  
**実験**: Walker Actuator Domain Randomization  
**比較モデル**: Baseline, DR, Model C, Oracle

