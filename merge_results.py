#!/usr/bin/env python
"""
評価結果のマージスクリプト（Windows対応版）

results_seed*.csv を1つの results.csv にマージ
"""

import sys
from pathlib import Path

def main():
    print("="*70)
    print("結果ファイルのマージ")
    print("="*70)
    
    # カレントディレクトリ
    current_dir = Path(".")
    
    # 出力ファイル
    output_file = current_dir / "results.csv"
    
    # seed 0-4 のファイルを探す
    seed_files = []
    for seed in range(5):
        seed_file = current_dir / f"results_seed{seed}.csv"
        if seed_file.exists():
            seed_files.append((seed, seed_file))
            print(f"✓ 見つかりました: {seed_file.name}")
        else:
            print(f"⚠ 見つかりません: {seed_file.name}")
    
    if not seed_files:
        print("\nエラー: results_seed*.csv が見つかりません")
        sys.exit(1)
    
    print(f"\n{len(seed_files)} 個のファイルをマージします")
    print()
    
    # マージ処理
    total_lines = 0
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        # 最初のファイルからヘッダーを取得
        first_seed, first_file = seed_files[0]
        with open(first_file, "r", encoding="utf-8") as f:
            header = f.readline()
            out_f.write(header)
            print(f"✓ ヘッダー追加: {first_file.name}")
        
        # 各ファイルのデータを追加（ヘッダーをスキップ）
        for seed, seed_file in seed_files:
            with open(seed_file, "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]  # ヘッダーをスキップ
                out_f.writelines(lines)
                total_lines += len(lines)
                print(f"✓ {len(lines):4d} 行追加: {seed_file.name}")
    
    print()
    print("="*70)
    print("マージ完了！")
    print("="*70)
    print(f"出力ファイル: {output_file}")
    print(f"総データ行数: {total_lines} 行（ヘッダー除く）")
    print(f"期待値: 約 3000 行（4モデル × 5 seeds × 5 params × 30 episodes）")
    print()
    
    # サンプル表示
    print("データのサンプル（最初の5行）:")
    with open(output_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 6:  # ヘッダー + 5行
                print(f"  {line.rstrip()}")
            else:
                break
    
    print()
    print("次のステップ:")
    print("  python evaluate/analyze_results.py")


if __name__ == "__main__":
    main()

