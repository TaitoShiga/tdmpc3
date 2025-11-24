#!/bin/bash
# 評価結果のマージスクリプト
# 
# 各seedごとに生成された results_seed*.csv を1つの results.csv にマージ
#
# 使用方法: bash slurm_scripts/merge_eval_results.sh

echo "=============================================="
echo "Merging evaluation results..."
echo "=============================================="

# 出力ファイル
OUTPUT="results.csv"

# 既存のファイルがあれば削除
if [ -f "$OUTPUT" ]; then
    echo "Removing existing $OUTPUT..."
    rm "$OUTPUT"
fi

# ヘッダーを追加（results_seed0.csv から）
if [ -f "results_seed0.csv" ]; then
    head -1 results_seed0.csv > "$OUTPUT"
    echo "✓ Added header from results_seed0.csv"
else
    echo "Error: results_seed0.csv not found!"
    exit 1
fi

# 各seedのデータを追加（ヘッダーをスキップ）
for seed in 0 1 2 3 4; do
    file="results_seed${seed}.csv"
    if [ -f "$file" ]; then
        # ヘッダー行をスキップして追加
        tail -n +2 "$file" >> "$OUTPUT"
        lines=$(tail -n +2 "$file" | wc -l)
        echo "✓ Added $lines lines from $file"
    else
        echo "⚠ Warning: $file not found, skipping..."
    fi
done

# 結果の確認
total_lines=$(wc -l < "$OUTPUT")
data_lines=$((total_lines - 1))

echo ""
echo "=============================================="
echo "Merge completed!"
echo "=============================================="
echo "Output file: $OUTPUT"
echo "Total lines: $total_lines (1 header + $data_lines data)"
echo ""
echo "Expected: ~3600 data lines (4 models × 5 seeds × 5 params × 30 episodes)"
echo ""

# データのサンプルを表示
echo "Sample of merged data (first 10 lines):"
head -10 "$OUTPUT"

echo ""
echo "Next step:"
echo "  python evaluate/analyze_results.py"

