"""Domain Randomization検証スクリプト"""
import sys
import os

# tdmpc2ディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tdmpc2'))

import numpy as np
from dm_control import suite
from envs.tasks import pendulum

# カスタムタスクを登録
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)

def test_mass_randomization():
    """質量が本当にエピソードごとに変わっているか確認"""
    print("=== Domain Randomization検証 ===\n")
    
    # 環境を作成
    env = suite.load('pendulum', 'swingup_randomized', 
                     task_kwargs={'random': 42})
    
    # 20エピソードで質量を記録
    masses = []
    for ep in range(20):
        env.reset()
        # Physics内部のモデルから質量を取得
        mass = env.physics.model.body_mass[-1]  # 最後のボディが振り子の質量
        masses.append(mass)
        print(f"Episode {ep:2d}: mass = {mass:.4f}")
    
    print(f"\n質量の統計:")
    print(f"  範囲指定: [0.5, 2.5]")
    print(f"  実際の最小値: {np.min(masses):.4f}")
    print(f"  実際の最大値: {np.max(masses):.4f}")
    print(f"  平均値: {np.mean(masses):.4f} (期待値: 1.5)")
    print(f"  標準偏差: {np.std(masses):.4f}")
    print(f"  ユニーク数: {len(np.unique(masses))}/20")
    
    # 検証
    all_unique = len(np.unique(masses)) == len(masses)
    in_range = np.all((np.array(masses) >= 0.5) & (np.array(masses) <= 2.5))
    
    print(f"\n検証結果:")
    if len(np.unique(masses)) == 1:
        print("  ❌ FAILED: 全エピソードで質量が同じ")
        print("     → Domain Randomizationが機能していません")
        return False
    elif not all_unique:
        print(f"  ⚠️  WARNING: 重複あり ({len(masses) - len(np.unique(masses))}個)")
        print("     → ランダム性は確認できますが、衝突の可能性")
    
    if not in_range:
        print("  ❌ FAILED: 範囲外の質量が検出されました")
        return False
    
    print("  ✅ PASSED: エピソードごとに質量が変化しています")
    print("  ✅ PASSED: すべての質量が指定範囲内です")
    return True

if __name__ == '__main__':
    success = test_mass_randomization()
    exit(0 if success else 1)

