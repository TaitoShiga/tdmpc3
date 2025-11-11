"""全Domain Randomizationタスクの検証スクリプト"""
import sys
import os

# tdmpc2ディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tdmpc2'))

import numpy as np
from dm_control import suite
from envs.tasks import ball_in_cup, reacher, hopper, pendulum

# カスタムタスクを登録
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)

def test_task_randomization(domain, task, body_index, param_name, expected_range, num_episodes=20):
    """汎用的なDR検証関数"""
    print(f"\n{'='*60}")
    print(f"Testing: {domain}-{task}")
    print(f"{'='*60}\n")
    
    # 環境を作成
    env = suite.load(domain, task, task_kwargs={'random': 42})
    
    # エピソードごとにパラメータを記録
    params = []
    for ep in range(num_episodes):
        env.reset()
        # Physics内部のモデルからパラメータを取得
        param = env.physics.model.body_mass[body_index]
        params.append(param)
        print(f"Episode {ep:2d}: {param_name} = {param:.4f}")
    
    print(f"\n{param_name}の統計:")
    print(f"  範囲指定: {expected_range}")
    print(f"  実際の最小値: {np.min(params):.4f}")
    print(f"  実際の最大値: {np.max(params):.4f}")
    print(f"  平均値: {np.mean(params):.4f}")
    print(f"  標準偏差: {np.std(params):.4f}")
    print(f"  ユニーク数: {len(np.unique(params))}/{num_episodes}")
    
    # 検証
    min_val, max_val = expected_range
    all_unique = len(np.unique(params)) == len(params)
    in_range = np.all((np.array(params) >= min_val) & (np.array(params) <= max_val))
    
    print(f"\n検証結果:")
    if len(np.unique(params)) == 1:
        print("  ❌ FAILED: 全エピソードでパラメータが同じ")
        print("     → Domain Randomizationが機能していません")
        return False
    elif not all_unique:
        print(f"  ⚠️  WARNING: 重複あり ({len(params) - len(np.unique(params))}個)")
        print("     → ランダム性は確認できますが、衝突の可能性")
    
    if not in_range:
        print("  ❌ FAILED: 範囲外のパラメータが検出されました")
        return False
    
    print("  ✅ PASSED: エピソードごとにパラメータが変化しています")
    print("  ✅ PASSED: すべてのパラメータが指定範囲内です")
    return True


def main():
    """全タスクをテスト"""
    print("="*60)
    print("Domain Randomization 全タスク検証")
    print("="*60)
    
    results = {}
    
    # 1. Pendulum Swingup (既存)
    results['pendulum-swingup'] = test_task_randomization(
        domain='pendulum',
        task='swingup_randomized',
        body_index=-1,  # 最後のbody = pole
        param_name='pole mass',
        expected_range=(0.5, 2.5)
    )
    
    # 2. Ball-in-Cup Catch
    results['cup-catch'] = test_task_randomization(
        domain='ball_in_cup',
        task='catch_randomized',
        body_index=2,  # ball body
        param_name='ball mass',
        expected_range=(0.003, 0.015)
    )
    
    # 3. Reacher Three-Link Easy
    print(f"\n{'='*60}")
    print(f"Testing: reacher-three_easy_randomized")
    print(f"{'='*60}\n")
    env = suite.load('reacher', 'three_easy_randomized', task_kwargs={'random': 42})
    print("Testing multiple links (arm0, arm1, hand)...")
    for link_idx in range(1, 4):
        masses = []
        for ep in range(10):
            env.reset()
            mass = env.physics.model.body_mass[link_idx]
            masses.append(mass)
        print(f"  Link {link_idx-1}: min={np.min(masses):.4f}, max={np.max(masses):.4f}, unique={len(np.unique(masses))}/10")
    results['reacher-three_easy'] = len(np.unique(masses)) > 1
    
    # 4. Hopper Stand
    results['hopper-stand'] = test_task_randomization(
        domain='hopper',
        task='stand_randomized',
        body_index=1,  # torso body
        param_name='torso mass',
        expected_range=(2.0, 6.0)
    )
    
    # 最終結果サマリ
    print(f"\n{'='*60}")
    print("最終結果サマリ")
    print(f"{'='*60}\n")
    
    for task, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {task}")
    
    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 全タスクが正常に動作しています！")
    else:
        print("⚠️  一部のタスクに問題があります")
    print(f"{'='*60}\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())

