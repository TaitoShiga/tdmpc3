"""
Oracle + Domain Randomization の統合テスト

DRで質量がランダム化された環境でも、Oracleが正しく
真の物理パラメータを取得できることを確認する。

使用方法:
    python test_oracle_with_dr.py
"""
import sys
import os

# tdmpc2ディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tdmpc2'))

os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from termcolor import colored
from dm_control import suite
from envs.tasks import pendulum

# カスタムタスクを登録
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)


class TestConfig:
    """テスト用の設定"""
    def __init__(self):
        self.task = 'pendulum-swingup-randomized'
        self.obs = 'state'
        self.seed = 0
        self.c_phys_dim = 1
        self.phys_param_type = 'mass'
        self.phys_param_indices = None
        self.phys_param_normalization = 'standard'
        self.phys_param_default = None
        self.phys_param_scale = None


def test_dr_environment():
    """DRの環境で質量がランダム化されることを確認"""
    print(colored('\n[Test 1] Domain Randomization Environment', 'cyan', attrs=['bold']))
    
    env = suite.load('pendulum', 'swingup_randomized', task_kwargs={'random': 42})
    
    masses = []
    for ep in range(10):
        env.reset()
        mass = env.physics.model.body_mass[-1]
        masses.append(mass)
        print(f"  Episode {ep}: mass = {mass:.4f}")
    
    masses = np.array(masses)
    print(f"\n  ✓ Mass statistics:")
    print(f"    - min:  {masses.min():.4f}")
    print(f"    - max:  {masses.max():.4f}")
    print(f"    - mean: {masses.mean():.4f}")
    print(f"    - std:  {masses.std():.4f}")
    
    # 質量が変動していることを確認
    assert masses.std() > 0.1, "Mass should vary across episodes"
    assert masses.min() >= 0.4, f"Mass too small: {masses.min()}"
    assert masses.max() <= 2.6, f"Mass too large: {masses.max()}"
    
    print(colored('  ✓ Test passed! Mass varies across episodes.', 'green'))
    return env


def test_physics_wrapper_with_dr():
    """DRの環境でPhysicsWrapperが正しく動作することを確認"""
    print(colored('\n[Test 2] Physics Wrapper with DR', 'cyan', attrs=['bold']))
    
    from envs import make_env
    from envs.wrappers.physics_param import wrap_with_physics_param
    
    cfg = TestConfig()
    env = make_env(cfg)
    env = wrap_with_physics_param(env, cfg)
    
    print(f"  ✓ Wrapper created")
    print(f"    - c_phys_dim: {env.c_phys_dim}")
    print(f"    - normalization: {env.normalization}")
    
    # 複数エピソードで物理パラメータを取得
    masses_raw = []
    masses_normalized = []
    
    for ep in range(10):
        obs = env.reset()
        c_phys = env.current_c_phys
        
        # 生の質量を取得
        mass_raw = env.env.unwrapped.physics.model.body_mass[-1]
        masses_raw.append(mass_raw)
        masses_normalized.append(c_phys[0].item())
        
        print(f"  Episode {ep}: raw_mass = {mass_raw:.4f}, c_phys = {c_phys[0]:.4f}")
    
    masses_raw = np.array(masses_raw)
    masses_normalized = np.array(masses_normalized)
    
    print(f"\n  ✓ Raw mass statistics:")
    print(f"    - min:  {masses_raw.min():.4f}")
    print(f"    - max:  {masses_raw.max():.4f}")
    print(f"    - mean: {masses_raw.mean():.4f}")
    
    print(f"\n  ✓ Normalized c_phys statistics:")
    print(f"    - min:  {masses_normalized.min():.4f}")
    print(f"    - max:  {masses_normalized.max():.4f}")
    print(f"    - mean: {masses_normalized.mean():.4f}")
    
    # 正規化が正しく行われているか確認
    # standard正規化: (x - 1.0) / 1.0
    expected_normalized = (masses_raw - 1.0) / 1.0
    np.testing.assert_allclose(masses_normalized, expected_normalized, rtol=1e-5)
    
    print(colored('  ✓ Test passed! Wrapper correctly extracts physics parameters from DR.', 'green'))
    return env


def test_c_phys_consistency_within_episode():
    """エピソード中は物理パラメータが一定であることを確認"""
    print(colored('\n[Test 3] c_phys Consistency within Episode', 'cyan', attrs=['bold']))
    
    from envs import make_env
    from envs.wrappers.physics_param import wrap_with_physics_param
    
    cfg = TestConfig()
    env = make_env(cfg)
    env = wrap_with_physics_param(env, cfg)
    
    for ep in range(3):
        obs = env.reset()
        c_phys_initial = env.current_c_phys.clone()
        
        print(f"\n  Episode {ep}: initial c_phys = {c_phys_initial[0]:.4f}")
        
        # エピソード中に複数ステップ実行
        for step in range(50):
            action = env.rand_act()
            obs, reward, done, info = env.step(action)
            c_phys_current = env.current_c_phys
            
            # エピソード中は同じ値のはず
            assert torch.allclose(c_phys_current, c_phys_initial), \
                f"c_phys changed within episode at step {step}"
            
            if done:
                break
        
        print(f"    - Ran {step+1} steps, c_phys remained constant")
    
    print(colored('  ✓ Test passed! c_phys is constant within episodes.', 'green'))


def test_oracle_components_with_dr():
    """Oracle版のコンポーネントがDRと統合できることを確認"""
    print(colored('\n[Test 4] Oracle Components with DR', 'cyan', attrs=['bold']))
    
    from envs import make_env
    from envs.wrappers.physics_param import wrap_with_physics_param
    from common.buffer_oracle import OracleBuffer
    from common.world_model_oracle import OracleWorldModel
    from tensordict import TensorDict
    
    cfg = TestConfig()
    cfg.obs_shape = {'state': (3,)}
    cfg.action_dim = 1
    cfg.episode_length = 500
    cfg.batch_size = 32
    cfg.horizon = 3
    cfg.buffer_size = 10000
    cfg.steps = 10000
    
    # 環境
    env = make_env(cfg)
    env = wrap_with_physics_param(env, cfg)
    
    # モデル
    cfg.latent_dim = 64
    cfg.mlp_dim = 128
    cfg.num_q = 2
    cfg.num_enc_layers = 1
    cfg.enc_dim = 64
    cfg.task_dim = 0
    cfg.dropout = 0.0
    cfg.simnorm_dim = 8
    cfg.multitask = False
    cfg.num_bins = 101
    cfg.vmin = -10
    cfg.vmax = 10
    
    model = OracleWorldModel(cfg).cuda()
    buffer = OracleBuffer(cfg)
    
    print(f"  ✓ Components created")
    
    # 複数エピソードでデータ収集
    print(f"\n  Collecting data from DR environment...")
    for ep in range(5):
        obs = env.reset()
        c_phys = env.current_c_phys
        mass_raw = env.env.unwrapped.physics.model.body_mass[-1]
        
        print(f"    Episode {ep}: mass = {mass_raw:.4f}, c_phys = {c_phys[0]:.4f}")
        
        episode_data = []
        for t in range(50):
            action = env.rand_act()
            next_obs, reward, done, info = env.step(action)
            
            td = TensorDict({
                'obs': obs.unsqueeze(0).cpu(),
                'action': action.unsqueeze(0).cpu(),
                'reward': torch.tensor([reward]),
                'terminated': torch.tensor([0.0]),
                'c_phys': c_phys.unsqueeze(0).cpu(),
            }, batch_size=(1,))
            episode_data.append(td)
            
            obs = next_obs
            if done:
                break
        
        buffer.add(torch.cat(episode_data))
    
    print(f"  ✓ Collected {buffer.num_eps} episodes")
    
    # バッファからサンプル
    obs_sample, action_sample, reward_sample, terminated_sample, task_sample, c_phys_sample = buffer.sample()
    
    print(f"\n  ✓ Sampled batch:")
    print(f"    - c_phys shape: {c_phys_sample.shape}")
    print(f"    - c_phys values (first 5): {c_phys_sample[:5, 0]}")
    
    # モデルでフォワードパス
    batch_size = c_phys_sample.shape[0]
    obs_test = obs_sample[0]  # (batch, obs_dim)
    action_test = action_sample[0]  # (batch, action_dim)
    
    z = model.encode(obs_test.cuda(), task=None)
    z_next = model.next(z, action_test.cuda(), task=None, c_phys=c_phys_sample.cuda())
    reward_pred = model.reward(z, action_test.cuda(), task=None, c_phys=c_phys_sample.cuda())
    
    print(f"\n  ✓ Forward pass successful:")
    print(f"    - z shape: {z.shape}")
    print(f"    - z_next shape: {z_next.shape}")
    print(f"    - reward_pred shape: {reward_pred.shape}")
    
    print(colored('  ✓ Test passed! Oracle components work with DR.', 'green'))


def main():
    """メインテスト関数"""
    print(colored('='*70, 'yellow', attrs=['bold']))
    print(colored('Oracle + Domain Randomization Integration Test', 'yellow', attrs=['bold']))
    print(colored('='*70, 'yellow', attrs=['bold']))
    
    try:
        # Test 1: DRの環境
        test_dr_environment()
        
        # Test 2: PhysicsWrapperとDRの統合
        test_physics_wrapper_with_dr()
        
        # Test 3: エピソード中の一貫性
        test_c_phys_consistency_within_episode()
        
        # Test 4: Oracleコンポーネントとの統合
        test_oracle_components_with_dr()
        
        print(colored('\n' + '='*70, 'green', attrs=['bold']))
        print(colored('✓ All tests passed!', 'green', attrs=['bold']))
        print(colored('='*70, 'green', attrs=['bold']))
        
        print(colored('\n📋 確認事項:', 'cyan', attrs=['bold']))
        print(colored('  ✅ DRで質量がエピソードごとにランダム化される', 'white'))
        print(colored('  ✅ PhysicsWrapperが正しく真の質量を取得できる', 'white'))
        print(colored('  ✅ エピソード中は物理パラメータが一定', 'white'))
        print(colored('  ✅ Oracleコンポーネントが正しく動作する', 'white'))
        
        print(colored('\n🚀 実行コマンド:', 'cyan', attrs=['bold']))
        print(colored('  # ベースライン（DRなし）', 'white'))
        print(colored('  python tdmpc2/train.py task=pendulum-swingup seed=0 steps=500000', 'yellow'))
        
        print(colored('\n  # ベースライン（DRあり）', 'white'))
        print(colored('  python tdmpc2/train.py task=pendulum-swingup-randomized seed=0 steps=500000', 'yellow'))
        
        print(colored('\n  # Oracle（DRなし）', 'white'))
        print(colored('  python tdmpc2/train.py task=pendulum-swingup use_oracle=true seed=0 steps=500000', 'yellow'))
        
        print(colored('\n  # Oracle（DRあり）← これが重要！', 'white'))
        print(colored('  python tdmpc2/train.py task=pendulum-swingup-randomized use_oracle=true seed=0 steps=500000', 'yellow'))
        
    except Exception as e:
        print(colored(f'\n✗ Test failed with error:', 'red', attrs=['bold']))
        print(colored(f'  {str(e)}', 'red'))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

