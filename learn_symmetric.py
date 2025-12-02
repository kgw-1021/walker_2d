import gymnasium as gym
import numpy as np
import torch
import os
from gymnasium import spaces

# Stable Baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# ==================== 1. 설정 (Config) ====================
class Config:
    env_name = "Walker2d-v5"
    total_timesteps = 1_000_000
    
    # 병렬 처리 설정
    n_envs = 8  # 병렬 환경 개수 (CPU 코어 수에 맞춰 조정)
    
    # 위상 및 궤적 파라미터
    T_stride = 0.7
    hip_amplitude = 0.25
    hip_center = 0.0
    hip_offset = 0.0
    knee_amplitude = 0.5
    knee_center = 0.8
    knee_offset = -np.pi/2
    
    # 보상 가중치
    lambda_vel_init = 0.1
    lambda_vel_final = 1.0
    lambda_track = 2.0
    lambda_ctrl = 0.001
    
    # 커리큘럼 설정
    curriculum_timesteps = 500_000
    
    # 경로 설정
    log_dir = "./walker_phase_sac_logs/"

# 디렉토리 생성
os.makedirs(Config.log_dir, exist_ok=True)

# ==================== 2. 목표 궤적 생성기 ====================
class TrajectoryGenerator:
    def __init__(self, config):
        self.cfg = config
    
    def get_target_angles(self, phase):
        left_hip = self.cfg.hip_amplitude * np.sin(2 * np.pi * phase + self.cfg.hip_offset) + self.cfg.hip_center
        left_knee = self.cfg.knee_amplitude * np.sin(2 * np.pi * phase + self.cfg.knee_offset) + self.cfg.knee_center
        
        right_phase = (phase + 0.5) % 1.0
        right_hip = self.cfg.hip_amplitude * np.sin(2 * np.pi * right_phase + self.cfg.hip_offset) + self.cfg.hip_center
        right_knee = self.cfg.knee_amplitude * np.sin(2 * np.pi * right_phase + self.cfg.knee_offset) + self.cfg.knee_center
        
        return {
            'left_hip': left_hip, 'left_knee': left_knee,
            'right_hip': right_hip, 'right_knee': right_knee
        }

# ==================== 3. 환경 래퍼 ====================
class PhaseAugmentedWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.cfg = config
        self.traj_gen = TrajectoryGenerator(config)
        self.joint_indices = {'left_hip': 3, 'left_knee': 4, 'right_hip': 6, 'right_knee': 7}
        
        self.phase = 0.0
        self.time = 0.0
        self.dt = 0.008
        self.current_vel_weight = self.cfg.lambda_vel_init
        
        # Observation Space 확장 (기존 + sin + cos + 4 errors)
        base_dim = self.env.observation_space.shape[0]
        new_dim = base_dim + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32
        )
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.phase = 0.0
        self.time = 0.0
        return self._augment_observation(obs), info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.time += self.dt
        self.phase = (self.time / self.cfg.T_stride) % 1.0
        
        aug_obs = self._augment_observation(obs)
        custom_reward, tracking_error = self._compute_reward(action, info)
        
        # 모니터링을 위해 info에 기록
        info['tracking_error'] = tracking_error
        info['phase'] = self.phase
        info['vel_weight'] = self.current_vel_weight
        
        return aug_obs, custom_reward, terminated, truncated, info
    
    def _augment_observation(self, base_obs):
        phase_enc = np.array([np.sin(2*np.pi*self.phase), np.cos(2*np.pi*self.phase)])
        
        qpos = self.env.unwrapped.data.qpos
        curr = {name: qpos[idx] for name, idx in self.joint_indices.items()}
        targ = self.traj_gen.get_target_angles(self.phase)
        
        errors = np.array([
            curr['left_hip'] - targ['left_hip'],
            curr['left_knee'] - targ['left_knee'],
            curr['right_hip'] - targ['right_hip'],
            curr['right_knee'] - targ['right_knee']
        ])
        
        return np.concatenate([base_obs, phase_enc, errors]).astype(np.float32)

    def _compute_reward(self, action, info):
        x_vel = info.get('x_velocity', self.env.unwrapped.data.qvel[0])
        
        qpos = self.env.unwrapped.data.qpos
        curr = {name: qpos[idx] for name, idx in self.joint_indices.items()}
        targ = self.traj_gen.get_target_angles(self.phase)
        
        tracking_error_sq = sum([(curr[k] - targ[k])**2 for k in self.joint_indices])
        control_cost = np.sum(np.square(action))
        
        healthy_reward = 1.0
        
        reward = (
            self.current_vel_weight * x_vel
            + healthy_reward
            - self.cfg.lambda_track * tracking_error_sq
            - self.cfg.lambda_ctrl * control_cost
        )
        return reward, tracking_error_sq
    
    def set_velocity_weight(self, weight):
        self.current_vel_weight = weight

# ==================== 4. 유용한 콜백들 ====================

class SaveVecNormalizeCallback(BaseCallback):
    """최고 모델 저장 시 VecNormalize 통계를 함께 저장 (업로드된 코드 참조)"""
    def __init__(self, vec_normalize_env, save_path):
        super().__init__()
        self.vec_normalize_env = vec_normalize_env
        self.save_path = save_path
    
    def _on_step(self):
        # EvalCallback이 이 콜백을 호출할 때 저장 수행
        self.vec_normalize_env.save(self.save_path)
        if self.verbose > 0:
            print(f"VecNormalize 통계 저장 완료: {self.save_path}")
        return True

class CurriculumAndMonitorCallback(BaseCallback):
    """커리큘럼 적용 및 추가 지표(Tracking Error) 모니터링"""
    def __init__(self, config, verbose=0):
        super().__init__(verbose)
        self.cfg = config
        self.tracking_errors = []
    
    def _on_step(self) -> bool:
        # 1. 커리큘럼 업데이트
        progress = min(self.num_timesteps / self.cfg.curriculum_timesteps, 1.0)
        new_weight = self.cfg.lambda_vel_init + progress * (self.cfg.lambda_vel_final - self.cfg.lambda_vel_init)
        self.training_env.env_method("set_velocity_weight", new_weight)
        
        # 2. 로깅 (Tracking Error 등)
        infos = self.locals.get("infos", [])
        for info in infos:
            if 'tracking_error' in info:
                self.tracking_errors.append(info['tracking_error'])
                
        # Tensorboard 기록
        self.logger.record("curriculum/velocity_weight", new_weight)
        
        if len(self.tracking_errors) > 0 and self.n_calls % 1000 == 0:
            mean_error = np.mean(self.tracking_errors)
            self.logger.record("train/tracking_error_mean", mean_error)
            self.tracking_errors = [] # 버퍼 초기화
            
        return True

# ==================== 5. 메인 실행 루프 ====================
def make_env(rank, seed=0):
    """SubprocVecEnv를 위한 환경 생성 함수"""
    def _init():
        env = gym.make(Config.env_name)
        env = PhaseAugmentedWrapper(env, Config())
        # 시드 설정 (각 프로세스마다 다르게)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    print(f"학습 시작: {Config.env_name} (Parallel Envs: {Config.n_envs})")
    
    # 1. 학습 환경 (병렬 처리 + 정규화)
    train_env = SubprocVecEnv([make_env(i) for i in range(Config.n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. 평가 환경 (단일 환경 + 정규화)
    # 중요: 평가 환경은 training=False로 설정하여 통계를 업데이트하지 않음
    eval_env = DummyVecEnv([make_env(999)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
    
    # 3. 콜백 설정
    # (1) 정규화 통계 저장 콜백
    save_vec_norm_cb = SaveVecNormalizeCallback(train_env, os.path.join(Config.log_dir, "vec_normalize.pkl"))
    
    # (2) 평가 콜백 (최고 모델 발견 시 save_vec_norm_cb 실행)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(Config.log_dir, "best_model"),
        log_path=os.path.join(Config.log_dir, "eval_logs"),
        eval_freq=2000,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=save_vec_norm_cb,
        verbose=1
    )
    
    # (3) 커리큘럼 및 모니터링 콜백
    curriculum_cb = CurriculumAndMonitorCallback(Config())
    
    # 4. 모델 생성 (SAC)
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        ent_coef='auto',
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=os.path.join(Config.log_dir, "tensorboard")
    )

    # 5. 학습 실행
    model.learn(total_timesteps=Config.total_timesteps, callback=[eval_callback, curriculum_cb])
    
    # 6. 최종 저장
    model.save(os.path.join(Config.log_dir, "final_model"))
    train_env.save(os.path.join(Config.log_dir, "vec_normalize_final.pkl"))
    print("학습 및 저장 완료!")
    
    train_env.close()
    eval_env.close()

    # ==================== 6. 최종 테스트 (시각화) ====================
    print("\n 최종 모델 테스트 시작...")
    
    # 테스트 환경 생성 (렌더링 모드)
    # 주의: DummyVecEnv 사용 (렌더링 용이성)
    test_env = DummyVecEnv([lambda: gym.make(Config.env_name, render_mode="human")])
    # 래퍼 다시 적용
    test_env = DummyVecEnv([lambda: PhaseAugmentedWrapper(gym.make(Config.env_name, render_mode="human"), Config())])
    
    # 정규화 통계 로드 (필수!)
    # best_model이 저장된 시점의 통계를 불러옵니다.
    stats_path = os.path.join(Config.log_dir, "vec_normalize.pkl")
    if os.path.exists(stats_path):
        test_env = VecNormalize.load(stats_path, test_env)
        test_env.training = False # 테스트 시 업데이트 중지
        test_env.norm_reward = False
        print("Loaded VecNormalize stats.")
    else:
        print("Warning: VecNormalize stats not found. Performance might be poor.")

    # 모델 로드
    model_path = os.path.join(Config.log_dir, "best_model", "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(Config.log_dir, "final_model.zip")
        
    model = SAC.load(model_path)
    
    # 실행 루프
    obs = test_env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
        # Subproc가 아니므로 바로 render 가능하지만, 
        # human mode로 make 했으므로 자동으로 창이 뜸
        test_env.envs[0].render()
        
        if done[0]:
            obs = test_env.reset()

    test_env.close()