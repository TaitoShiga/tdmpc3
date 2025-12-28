"""
Hopper Hop Backwards の thigh_length 実装テスト
"""

import numpy as np
from dm_control import suite

# カスタムタスクを登録
from tdmpc2.envs.tasks import hopper

def test_fixed_length_tasks():
    """固定長タスクの動作確認"""
    print("=" * 80)
    print("Test 1: 固定長タスクの動作確認")
    print("=" * 80)
    
    lengths = [
        (0.25, '076x'),
        (0.33, '10x'),
        (0.39, '12x'),
        (0.43, '13x'),
        (0.45, '136x'),
    ]
    
    for expected_length, suffix in lengths:
        task_name = f'hopper-hop_backwards_length_{suffix}'
        print(f"\n[{task_name}]")
        
        try:
            env = suite.load('hopper', f'hop_backwards_length_{suffix}')
            time_step = env.reset()
            
            # タスクから thigh_length を取得
            task = env.task
            if hasattr(task, 'current_thigh_length'):
                actual_length = task.current_thigh_length
                print(f"  ✓ current_thigh_length = {actual_length:.4f} (expected: {expected_length:.4f})")
                assert abs(actual_length - expected_length) < 1e-6, "Length mismatch!"
            else:
                print(f"  ✗ Task does not have current_thigh_length attribute")
            
            # 数ステップ実行
            for _ in range(5):
                action = np.random.uniform(-1, 1, env.action_spec().shape)
                time_step = env.step(action)
            
            print(f"  ✓ Episode runs successfully (reward: {time_step.reward:.4f})")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")


def test_randomized_task():
    """DR版タスクの動作確認"""
    print("\n" + "=" * 80)
    print("Test 2: Domain Randomization版タスクの動作確認")
    print("=" * 80)
    
    try:
        env = suite.load('hopper', 'hop_backwards_randomized')
        
        lengths = []
        for ep in range(5):
            time_step = env.reset()
            
            # タスクから thigh_length を取得
            task = env.task
            if hasattr(task, 'current_thigh_length'):
                length = task.current_thigh_length
                lengths.append(length)
                print(f"\n  Episode {ep}: thigh_length = {length:.4f}")
            else:
                print(f"\n  Episode {ep}: Task does not have current_thigh_length")
            
            # 数ステップ実行
            total_reward = 0
            for _ in range(10):
                action = np.random.uniform(-1, 1, env.action_spec().shape)
                time_step = env.step(action)
                total_reward += time_step.reward
            
            print(f"    10-step reward: {total_reward:.4f}")
        
        # 統計
        if lengths:
            print(f"\n  Length statistics:")
            print(f"    Min:  {min(lengths):.4f}")
            print(f"    Max:  {max(lengths):.4f}")
            print(f"    Mean: {np.mean(lengths):.4f}")
            print(f"    Std:  {np.std(lengths):.4f}")
            
            # 範囲チェック
            assert min(lengths) >= 0.25, "Length too small!"
            assert max(lengths) <= 0.45, "Length too large!"
            print(f"  ✓ All lengths within expected range [0.25, 0.45]")
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_physics_param_wrapper():
    """PhysicsParamWrapper との統合テスト"""
    print("\n" + "=" * 80)
    print("Test 3: PhysicsParamWrapper 統合テスト")
    print("=" * 80)
    
    try:
        from tdmpc2.envs.wrappers.physics_param import PhysicsParamWrapper
        import gymnasium as gym
        from dm_control import suite
        from shimmy import DmControlCompatibilityV0
        
        # 環境作成
        dm_env = suite.load('hopper', 'hop_backwards_randomized')
        env = DmControlCompatibilityV0(dm_env, render_mode=None)
        
        # Wrapperを適用
        wrapped_env = PhysicsParamWrapper(
            env=env,
            param_type='length',
            domain='hopper',
            task='hop_backwards'
        )
        
        print(f"  c_phys_dim: {wrapped_env.c_phys_dim}")
        
        # 複数エピソードで確認
        for ep in range(3):
            obs = wrapped_env.reset()
            
            # 物理パラメータ取得
            c_phys = wrapped_env.current_c_phys
            
            # タスクからの真値
            task = wrapped_env.env.unwrapped.task
            if hasattr(task, 'current_thigh_length'):
                true_length = task.current_thigh_length
                
                # 正規化されたc_physを逆変換
                denorm_length = (c_phys[0].item() * wrapped_env.scale[0] + 
                                wrapped_env.default_value[0])
                
                print(f"\n  Episode {ep}:")
                print(f"    True length:        {true_length:.4f}")
                print(f"    Normalized c_phys:  {c_phys[0]:.4f}")
                print(f"    Denormalized:       {denorm_length:.4f}")
                
                # 一致確認（誤差許容）
                if abs(denorm_length - true_length) < 0.01:
                    print(f"    ✓ Match!")
                else:
                    print(f"    ✗ Mismatch! (diff: {abs(denorm_length - true_length):.4f})")
        
        print("\n  ✓ PhysicsParamWrapper integration successful")
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_fixed_length_tasks()
    test_randomized_task()
    test_physics_param_wrapper()
    
    print("\n" + "=" * 80)
    print("全テスト完了")
    print("=" * 80)

