"""
Model O (Oracle) のクイック動作確認スクリプト

小規模な実験で実装が正常に動作するか確認する。

使用方法:
    python test_oracle_quick.py
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

# 設定の作成
class QuickTestConfig:
    """クイックテスト用の最小限の設定"""
    def __init__(self):
        # Environment
        self.task = 'pendulum-swingup'
        self.obs = 'state'
        self.episodic = False
        
        # Training
        self.steps = 1000  # 短い学習
        self.seed = 0
        self.batch_size = 32  # 小さいバッチ
        self.buffer_size = 10000
        self.seed_steps = 200
        self.eval_episodes = 2
        self.eval_freq = 500
        self.log_interval = 100
        
        # Model
        self.model_size = 1  # 最小モデル
        self.latent_dim = 64
        self.mlp_dim = 128
        self.num_q = 2
        self.num_enc_layers = 1
        self.enc_dim = 64
        self.task_dim = 0
        self.dropout = 0.0
        self.simnorm_dim = 8
        
        # Oracle specific
        self.c_phys_dim = 1
        self.phys_param_type = 'mass'
        self.phys_param_indices = None
        self.phys_param_normalization = 'standard'
        self.phys_param_default = None
        self.phys_param_scale = None
        
        # Planning
        self.mpc = True
        self.horizon = 3
        self.num_samples = 128  # 小さい
        self.num_elites = 16
        self.num_pi_trajs = 8
        self.iterations = 3  # 少ない
        self.min_std = 0.05
        self.max_std = 2
        self.temperature = 0.5
        
        # Optimization
        self.lr = 1e-3
        self.enc_lr_scale = 0.3
        self.grad_clip_norm = 10
        self.tau = 0.01
        self.rho = 0.5
        
        # Loss coefficients
        self.consistency_coef = 2
        self.reward_coef = 0.1
        self.value_coef = 0.1
        self.termination_coef = 1
        
        # Actor
        self.log_std_min = -10
        self.log_std_max = 2
        self.entropy_coef = 1e-4
        
        # Critic
        self.num_bins = 101
        self.vmin = -10
        self.vmax = 10
        
        # Discount
        self.discount_denom = 5
        self.discount_min = 0.95
        self.discount_max = 0.995
        
        # Misc
        self.compile = False
        self.save_video = False
        self.save_agent = False
        self.multitask = False
        self.work_dir = './test_oracle_output'


def test_physics_wrapper():
    """物理パラメータWrapperのテスト"""
    print(colored('\n[Test 1] Physics Parameter Wrapper', 'cyan', attrs=['bold']))
    
    from envs import make_env
    from envs.wrappers.physics_param import wrap_with_physics_param
    
    cfg = QuickTestConfig()
    env = make_env(cfg)
    env = wrap_with_physics_param(env, cfg)
    
    print(f"  ✓ Wrapper created")
    print(f"    - c_phys_dim: {env.c_phys_dim}")
    print(f"    - normalization: {env.normalization}")
    print(f"    - default_value: {env.default_value}")
    
    # リセットして物理パラメータを取得
    obs = env.reset()
    c_phys = env.current_c_phys
    
    print(f"  ✓ Physics parameter extracted")
    print(f"    - c_phys shape: {c_phys.shape}")
    print(f"    - c_phys value: {c_phys}")
    
    assert c_phys.shape == (cfg.c_phys_dim,), f"Expected shape ({cfg.c_phys_dim},), got {c_phys.shape}"
    assert torch.is_tensor(c_phys), "c_phys should be a tensor"
    
    print(colored('  ✓ Test passed!', 'green'))
    return env


def test_oracle_buffer(env):
    """Oracle用Bufferのテスト"""
    print(colored('\n[Test 2] Oracle Buffer', 'cyan', attrs=['bold']))
    
    from common.buffer_oracle import OracleBuffer
    from tensordict import TensorDict
    
    cfg = QuickTestConfig()
    buffer = OracleBuffer(cfg)
    
    print(f"  ✓ Buffer created")
    print(f"    - capacity: {buffer.capacity:,}")
    print(f"    - c_phys_dim: {buffer.c_phys_dim}")
    
    # ダミーエピソードを追加
    episode_length = 100
    obs = env.reset()
    c_phys = env.current_c_phys
    
    episode_data = []
    for t in range(episode_length):
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
    
    episode_td = torch.cat(episode_data)
    buffer.add(episode_td)
    
    print(f"  ✓ Episode added to buffer")
    print(f"    - episode length: {len(episode_data)}")
    print(f"    - num episodes: {buffer.num_eps}")
    
    # サンプリングテスト（複数エピソード追加後）
    for _ in range(4):
        buffer.add(episode_td)
    
    obs_sample, action_sample, reward_sample, terminated_sample, task_sample, c_phys_sample = buffer.sample()
    
    print(f"  ✓ Batch sampled")
    print(f"    - obs shape: {obs_sample.shape}")
    print(f"    - c_phys shape: {c_phys_sample.shape}")
    
    assert c_phys_sample.shape == (cfg.batch_size, cfg.c_phys_dim), \
        f"Expected c_phys shape ({cfg.batch_size}, {cfg.c_phys_dim}), got {c_phys_sample.shape}"
    
    print(colored('  ✓ Test passed!', 'green'))
    return buffer


def test_oracle_world_model():
    """Oracle用WorldModelのテスト"""
    print(colored('\n[Test 3] Oracle WorldModel', 'cyan', attrs=['bold']))
    
    from common.world_model_oracle import OracleWorldModel
    
    cfg = QuickTestConfig()
    cfg.obs_shape = {'state': (3,)}
    cfg.action_dim = 1
    
    model = OracleWorldModel(cfg).cuda()
    
    print(f"  ✓ WorldModel created")
    print(f"    - total params: {model.total_params:,}")
    print(f"    - c_phys_dim: {model.c_phys_dim}")
    
    # ダミー入力でテスト
    batch_size = 16
    obs = torch.randn(batch_size, 3).cuda()
    action = torch.randn(batch_size, 1).cuda()
    c_phys = torch.randn(batch_size, 1).cuda()
    
    # Encode
    z = model.encode(obs, task=None)
    print(f"  ✓ Encode: {obs.shape} -> {z.shape}")
    
    # Next
    z_next = model.next(z, action, task=None, c_phys=c_phys)
    print(f"  ✓ Next: {z.shape} -> {z_next.shape}")
    
    # Reward
    reward = model.reward(z, action, task=None, c_phys=c_phys)
    print(f"  ✓ Reward: -> {reward.shape}")
    
    # Q
    q = model.Q(z, action, task=None, c_phys=c_phys, return_type='min')
    print(f"  ✓ Q: -> {q.shape}")
    
    # Pi
    action_sample, info = model.pi(z, task=None, c_phys=c_phys)
    print(f"  ✓ Pi: -> {action_sample.shape}")
    
    print(colored('  ✓ Test passed!', 'green'))
    return model


def test_oracle_agent(env):
    """Oracle版TDMPC2のテスト"""
    print(colored('\n[Test 4] Oracle TDMPC2 Agent', 'cyan', attrs=['bold']))
    
    from tdmpc2_oracle import TDMPC2Oracle
    
    cfg = QuickTestConfig()
    cfg.obs_shape = {'state': env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = 500
    
    agent = TDMPC2Oracle(cfg)
    
    print(f"  ✓ Agent created")
    
    # act()のテスト
    obs = env.reset()
    c_phys = env.current_c_phys
    
    action = agent.act(obs, c_phys, t0=True, eval_mode=True)
    
    print(f"  ✓ Action selected")
    print(f"    - action shape: {action.shape}")
    print(f"    - action value: {action}")
    
    assert action.shape == (cfg.action_dim,), f"Expected action shape ({cfg.action_dim},), got {action.shape}"
    
    print(colored('  ✓ Test passed!', 'green'))
    return agent


def test_mini_training(env):
    """ミニ学習ループのテスト"""
    print(colored('\n[Test 5] Mini Training Loop', 'cyan', attrs=['bold']))
    
    from common.buffer_oracle import OracleBuffer
    from tdmpc2_oracle import TDMPC2Oracle
    from tensordict import TensorDict
    
    cfg = QuickTestConfig()
    cfg.obs_shape = {'state': env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = 500
    
    agent = TDMPC2Oracle(cfg)
    buffer = OracleBuffer(cfg)
    
    # シードデータの収集
    print("  Collecting seed data...")
    for ep in range(5):
        obs = env.reset()
        c_phys = env.current_c_phys
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
    
    # 学習ループ
    print("  Running mini training...")
    losses = []
    for step in range(10):
        info = agent.update(buffer)
        losses.append(info['total_loss'].item())
        if step % 5 == 0:
            print(f"    Step {step}: loss = {losses[-1]:.4f}")
    
    print(f"  ✓ Training completed")
    print(f"    - initial loss: {losses[0]:.4f}")
    print(f"    - final loss: {losses[-1]:.4f}")
    
    # 推論テスト
    obs = env.reset()
    c_phys = env.current_c_phys
    action = agent.act(obs, c_phys, t0=True, eval_mode=True)
    
    print(f"  ✓ Inference after training: action = {action}")
    
    print(colored('  ✓ Test passed!', 'green'))


def main():
    """メインテスト関数"""
    print(colored('='*70, 'yellow', attrs=['bold']))
    print(colored('Model O (Oracle) Quick Test', 'yellow', attrs=['bold']))
    print(colored('='*70, 'yellow', attrs=['bold']))
    
    try:
        # Test 1: Physics Wrapper
        env = test_physics_wrapper()
        
        # Test 2: Oracle Buffer
        buffer = test_oracle_buffer(env)
        
        # Test 3: Oracle WorldModel
        model = test_oracle_world_model()
        
        # Test 4: Oracle Agent
        agent = test_oracle_agent(env)
        
        # Test 5: Mini Training
        test_mini_training(env)
        
        print(colored('\n' + '='*70, 'green', attrs=['bold']))
        print(colored('✓ All tests passed!', 'green', attrs=['bold']))
        print(colored('='*70, 'green', attrs=['bold']))
        print(colored('\n次のステップ:', 'cyan', attrs=['bold']))
        print(colored('  1. 本格的な学習を開始:', 'white'))
        print(colored('     python tdmpc2/train_oracle.py task=pendulum-swingup seed=0', 'yellow'))
        print(colored('  2. 複数シードで実行:', 'white'))
        print(colored('     for seed in 0 1 2; do python tdmpc2/train_oracle.py seed=$seed; done', 'yellow'))
        
    except Exception as e:
        print(colored(f'\n✗ Test failed with error:', 'red', attrs=['bold']))
        print(colored(f'  {str(e)}', 'red'))
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

