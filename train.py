import os
import psutil
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# 导入你的自定义环境
from elytra_env import ElytraEnv

CONFIG = { 
    'max_steps': 500, 
    'yaw_scale': 50.0, 
    'firework_threshold': 0.5, 
    'success_distance': 0.50, 
}

# ==========================================
# 1. 自定义 Callback 用于扩展日志
# ==========================================
class DetailedLogCallback(BaseCallback):
    """
    自定义回调函数，用于在训练日志中添加 ep_rew_max 和 ep_rew_min。
    SB3 默认只记录 ep_rew_mean。
    """
    def __init__(self, verbose=0):
        super(DetailedLogCallback, self).__init__(verbose)
        self.current_successes = 0
        self.current_episodes = 0
        
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        
        for info in infos:
            if 'terminal_observation' in info or info.get('TimeLimit.truncated', False):
                self.current_episodes += 1
                
                if info.get('episode_success', False):
                    self.current_successes += 1
                    
        return True

    def _on_rollout_end(self) -> None:
        ep_infos = self.model.ep_info_buffer
        
        if ep_infos and len(ep_infos) > 0:
            rewards = [info['r'] for info in ep_infos]
            
            self.logger.record("rollout/ep_rew_max", np.max(rewards))
            self.logger.record("rollout/ep_rew_min", np.min(rewards))
            
            if self.current_episodes > 0:
                success_rate = (self.current_successes / self.current_episodes) * 100
                self.logger.record("rollout/success_rate", success_rate)
            
            self.current_successes = 0
            self.current_episodes = 0

        actions = self.locals.get('clipped_actions')
        if actions is not None:
            min_dpitch, min_dyaw, _   = np.amin(actions.T, axis=1)
            max_dpitch, max_dyaw, _   = np.amax(actions.T, axis=1)
            mean_dpitch, mean_dyaw, _ = np.mean(actions.T, axis=1)
            self.logger.record("action/min/dpitch", min_dpitch)
            self.logger.record("action/max/dpitch", max_dpitch)
            self.logger.record("action/mean/dpitch", mean_dpitch)
            self.logger.record("action/min/dyaw", min_dyaw)
            self.logger.record("action/max/dyaw", max_dyaw)
            self.logger.record("action/mean/dyaw", mean_dyaw)
            
# ==========================================
# 2. 环境构建函数
# ==========================================
def make_env(rank, seed=0):
    """
    用于创建环境的工厂函数，用于 Vectorized Environment。
    """
    def _init():
        # 训练配置：关闭渲染以加快速度
        config = CONFIG.copy()
        config.update(render_mode=None) # 训练时不显示 3D 窗口
        env = ElytraEnv(config=config)
        env = Monitor(env) 
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def make_eval_env(rank, seed=0):
    def _init():
        config = CONFIG.copy()
        config.update(render_mode=None)
        env = ElytraEnv(config=config)
        env = Monitor(env) 
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init
    
# ==========================================
# 3. 主训练流程
# ==========================================
def set_priority():
    current_process = psutil.Process(os.getpid())

    if os.name == 'nt':
        current_process.nice(psutil.REALTIME_PRIORITY_CLASS)

    else:
        current_process.nice(-10)

if __name__ == "__main__":
    set_priority()
    
    # --- 参数设置 ---
    TOTAL_TIMESTEPS = 5_000_000   # 总训练步数
    NUM_CPU = 12
    
    EVAL_FREQ = 500_000
    SAVE_FREQ = 1_000_000
    # EVAL_FREQ = 10_000
    
    # 路径设置
    log_dir = "./logs/"
    model_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Starting training with {NUM_CPU} environments ...")

    # --- 1. 创建向量化环境 (多进程加速) ---
    # 使用 DummyVecEnv 调试更方便，SubprocVecEnv 速度更快
    # 如果你在 Windows 上遇到多进程问题，请改用 DummyVecEnv
    env = SubprocVecEnv([make_env(i) for i in range(NUM_CPU)])
    
    print('Starting Evaluating with 1 environment ...')
    eval_env = SubprocVecEnv([make_eval_env(NUM_CPU)])
    # env = DummyVecEnv([make_env(i) for i in range(NUM_CPU)]) 

    # --- 2. 初始化 PPO 模型 ---
    model = PPO(
        'MlpPolicy',
        env,
        # learning_rate=0.0003,
        # learning_rate=0.00005,
        learning_rate=0.00015,
        n_steps=4096,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        # clip_range=0.2,
        # clip_range_vf=None,
        # normalize_advantage=True,
        ent_coef=0.01,
        # vf_coef=0.5,
        # max_grad_norm=0.5,
        # use_sde=False,
        # sde_sample_freq=-1,
        # rollout_buffer_class=None,
        # rollout_buffer_kwargs= None,
        target_kl=0.03,
        # stats_window_size=100,
        tensorboard_log=log_dir,
        # policy_kwargs=None,
        verbose=1,
        # seed=0,
        device="cuda:0",
        # _init_setup_model=True
    )
    
    print("load model ... ", end='')
    old_model_path = './models/free_yaw_0.75/best_model/best_model'
    
    old_model = PPO.load(old_model_path, device='cuda:0')
    model.policy.load_state_dict(old_model.policy.state_dict())
    model.num_timesteps = old_model.num_timesteps
    
    del old_model
    print('done.')

    # --- 3. 开始训练 ---
    # callback=DetailedLogCallback() 会自动把 max/min reward 注入到日志表格中
    # progress_bar=True 会显示总进度条
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[
                DetailedLogCallback(),
                EvalCallback(
                    eval_env=eval_env, 
                    eval_freq=max(EVAL_FREQ // NUM_CPU, 1), 
                    best_model_save_path=os.path.join(model_dir, 'best_model'), 
                    deterministic=True
                ), 
                CheckpointCallback(
                    save_freq=max(SAVE_FREQ // NUM_CPU, 1), 
                    save_path="./models/checkpoints/",  
                    name_prefix="elytra_ppo_free_yaw", 
                    verbose=1
                )
            ], 
            progress_bar=True, 
            reset_num_timesteps=False, 
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model...")

    # --- 4. 保存模型 ---
    model_path = os.path.join(model_dir, "elytra_ppo_free_yaw")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # 关闭环境
    eval_env.close()
