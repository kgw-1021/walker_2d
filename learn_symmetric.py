import gymnasium as gym
import numpy as np
import torch
import os
from gymnasium import spaces

# Stable Baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor  # [추가] 모니터 래퍼
from stable_baselines3.common.utils import set_random_seed

# ==================== 1. 설정 (Config) ====================
class Config:
    env_name = "Walker2d-v5"
    total_timesteps = 3_000_000
    
    # 병렬 처리 설정
    n_envs = 8
    
    # 위상 및 궤적 파라미터 (적응형)
    stride_length_base = 0.5    # 기본 보폭 (m)
    stride_length_coef = 0.2    # 속도 계수
    stride_time_base = 1.0      # 기본 주기 (s)
    stride_time_coef = -0.1     # 속도 계수
    x_vel_min = 0.1             # 최소 속도
    
    # 궤적 파라미터 (MuJoCo Walker2d 관절 범위 고려)
    hip_amplitude = 0.8
    hip_center = 0.0
    hip_offset = 0.0
    
    # Knee: Walker2d는 0(폄) ~ -150(굽힘)도 범위
    knee_amplitude = 0.7
    knee_center = -1.0          # 음수 중심
    knee_offset = -np.pi/2      
    
    # 보상 가중치
    lambda_vel_init = 0.1      
    lambda_vel_final = 1.0       
    lambda_track_init = 0.0     
    lambda_track_final = 0.5     
    lambda_ctrl = 0.001
    
    # 커리큘럼 설정
    curriculum_vel_steps = 1_000_000 
    curriculum_track_steps = 1_800_000
    
    # DUP (Mirror Augmentation) 설정
    enable_mirror_augmentation = True
    mirror_probability = 0.5
    
    # 경로 설정
    log_dir = "./walker_phase_sac_logs/"

os.makedirs(Config.log_dir, exist_ok=True)

# ==================== 2. Mirror 함수 (DUP 구현) ====================
class MirrorAugmentation:
    @staticmethod
    def mirror_obs(obs):
        mirrored = obs.copy()
        # 1. Root Angle (Pitch) 반전
        mirrored[1] = -obs[1]
        # 2. 관절 각도 Swap (Right <-> Left)
        mirrored[2:5] = obs[5:8]
        mirrored[5:8] = obs[2:5]
        # 3. Root Angular Velocity 반전
        mirrored[10] = -obs[10]
        # 4. 각속도 Swap
        mirrored[11:14] = obs[14:17]
        mirrored[14:17] = obs[11:14]
        return mirrored
    
    @staticmethod
    def mirror_action(action):
        mirrored = action.copy()
        mirrored[0:3] = action[3:6]
        mirrored[3:6] = action[0:3]
        return mirrored
    
    @staticmethod
    def mirror_obs_augmented(aug_obs):
        mirrored = aug_obs.copy()
        mirrored[:17] = MirrorAugmentation.mirror_obs(aug_obs[:17])
        # Tracking errors swap
        mirrored[19:21] = aug_obs[21:23]
        mirrored[21:23] = aug_obs[19:21]
        return mirrored

# ==================== 3. 목표 궤적 생성기 ====================
class TrajectoryGenerator:
    def __init__(self, config):
        self.cfg = config
    
    def get_target_angles(self, phase):
        # Right Leg (Base Phase)
        right_hip = self.cfg.hip_amplitude * np.sin(2 * np.pi * phase + self.cfg.hip_offset) + self.cfg.hip_center
        right_knee = self.cfg.knee_amplitude * np.sin(2 * np.pi * phase + self.cfg.knee_offset) + self.cfg.knee_center
        
        # Left Leg (Phase + 0.5)
        left_phase = (phase + 0.5) % 1.0
        left_hip = self.cfg.hip_amplitude * np.sin(2 * np.pi * left_phase + self.cfg.hip_offset) + self.cfg.hip_center
        left_knee = self.cfg.knee_amplitude * np.sin(2 * np.pi * left_phase + self.cfg.knee_offset) + self.cfg.knee_center
        
        # 물리적 제약
        right_knee = np.minimum(right_knee, 0.0)
        left_knee = np.minimum(left_knee, 0.0)
        
        return {
            'right_hip': right_hip, 'right_knee': right_knee,
            'left_hip': left_hip, 'left_knee': left_knee
        }

# ==================== 4. 환경 래퍼 ====================
class PhaseAugmentedWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.cfg = config
        self.traj_gen = TrajectoryGenerator(config)
        
        # MuJoCo qpos 인덱스
        self.joint_indices = {
            'right_hip': 3, 'right_knee': 4, 
            'left_hip': 6, 'left_knee': 7
        }
        
        self.phase = 0.0
        self.time = 0.0
        self.dt = 0.008
        self.current_vel_weight = self.cfg.lambda_vel_init
        self.current_track_weight = self.cfg.lambda_track_init
        self.use_mirror = False
        
        base_dim = self.env.observation_space.shape[0]
        new_dim = base_dim + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32
        )
    
    def _get_adaptive_stride_params(self, velocity):
        v = np.clip(abs(velocity), self.cfg.x_vel_min, 3.0)
        stride_length = self.cfg.stride_length_base + self.cfg.stride_length_coef * v
        stride_time = self.cfg.stride_time_base + self.cfg.stride_time_coef * v
        stride_time = max(stride_time, 0.3)
        return stride_length, stride_time
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.phase = 0.0
        self.time = 0.0
        
        if self.cfg.enable_mirror_augmentation:
            self.use_mirror = np.random.rand() < self.cfg.mirror_probability
        else:
            self.use_mirror = False
        
        aug_obs = self._augment_observation(obs)
        if self.use_mirror:
            aug_obs = MirrorAugmentation.mirror_obs_augmented(aug_obs)
        return aug_obs, info
        
    def step(self, action):
        actual_action = action
        if self.use_mirror:
            actual_action = MirrorAugmentation.mirror_action(action)
        
        obs, reward, terminated, truncated, info = self.env.step(actual_action)
        x_vel = info.get('x_velocity', self.env.unwrapped.data.qvel[0])

        # 보행 분석 파라미터 계산
        stride_length, stride_time = self._get_adaptive_stride_params(x_vel)
        cadence = 60.0 / stride_time if stride_time > 0 else 0.0
        
        self.time += self.dt
        delta_phi = self.dt / stride_time
        self.phase = (self.phase + delta_phi) % 1.0
        
        aug_obs = self._augment_observation(obs)
        custom_reward, tracking_error_L1 = self._compute_reward(actual_action, info)
        
        if self.use_mirror:
            aug_obs = MirrorAugmentation.mirror_obs_augmented(aug_obs)
        
        # [수정] 분석 정보를 info에 담기 (TensorBoard 기록용)
        info['tracking_error_L1'] = tracking_error_L1
        info['stride_length'] = stride_length
        info['stride_time'] = stride_time
        info['cadence'] = cadence
        info['phase'] = self.phase
        info['is_mirrored'] = self.use_mirror
        
        return aug_obs, custom_reward, terminated, truncated, info
    
    def _augment_observation(self, base_obs):
        phase_enc = np.array([np.sin(2*np.pi*self.phase), np.cos(2*np.pi*self.phase)])
        qpos = self.env.unwrapped.data.qpos
        curr = {name: qpos[idx] for name, idx in self.joint_indices.items()}
        targ = self.traj_gen.get_target_angles(self.phase)
        
        errors = np.array([
            curr['right_hip'] - targ['right_hip'],
            curr['right_knee'] - targ['right_knee'],
            curr['left_hip'] - targ['left_hip'],
            curr['left_knee'] - targ['left_knee']
        ])
        return np.concatenate([base_obs, phase_enc, errors]).astype(np.float32)

    def _compute_reward(self, action, info):
        x_vel = info.get('x_velocity', self.env.unwrapped.data.qvel[0])
        qpos = self.env.unwrapped.data.qpos
        curr = {name: qpos[idx] for name, idx in self.joint_indices.items()}
        targ = self.traj_gen.get_target_angles(self.phase)
        
        tracking_error_L1 = sum([abs(curr[k] - targ[k]) for k in self.joint_indices])
        control_cost = np.sum(np.square(action))
        healthy_reward = 5.0
        
        reward = (
            self.current_vel_weight * x_vel
            + healthy_reward
            - self.current_track_weight * tracking_error_L1
            - self.cfg.lambda_ctrl * control_cost
        )
        return reward, tracking_error_L1
    
    def set_velocity_weight(self, weight):
        self.current_vel_weight = weight

    def set_tracking_weight(self, weight):
        self.current_track_weight = weight

# ==================== 5. 콜백들 (분석 기능 복구) ====================

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, vec_normalize_env, save_path):
        super().__init__()
        self.vec_normalize_env = vec_normalize_env
        self.save_path = save_path
    
    def _on_step(self):
        self.vec_normalize_env.save(self.save_path)
        return True

class CurriculumAndMonitorCallback(BaseCallback):
    def __init__(self, config, verbose=0):
        super().__init__(verbose)
        self.cfg = config
        # 로깅을 위한 버퍼
        self.tracking_errors = []
        self.stride_lengths = []
        self.stride_times = []
        self.cadences = []
    
    def _on_step(self) -> bool:
        # 커리큘럼 업데이트
        progress_vel = min(self.num_timesteps / self.cfg.curriculum_vel_steps, 1.0)
        new_vel_weight = self.cfg.lambda_vel_init + progress_vel * (self.cfg.lambda_vel_final - self.cfg.lambda_vel_init)
        
        progress_track = max(0.0, (self.num_timesteps - self.cfg.curriculum_vel_steps) / (self.cfg.curriculum_track_steps - self.cfg.curriculum_vel_steps))
        progress_track = min(progress_track, 1.0)
        new_track_weight = self.cfg.lambda_track_init + progress_track * (self.cfg.lambda_track_final - self.cfg.lambda_track_init)

        self.training_env.env_method("set_velocity_weight", new_vel_weight)
        self.training_env.env_method("set_tracking_weight", new_track_weight)
        
        # 정보 수집
        infos = self.locals.get("infos", [])
        for info in infos:
            if 'tracking_error_L1' in info:
                self.tracking_errors.append(info['tracking_error_L1'])
            # [추가] 보행 분석 데이터 수집
            if 'stride_length' in info:
                self.stride_lengths.append(info['stride_length'])
            if 'stride_time' in info:
                self.stride_times.append(info['stride_time'])
            if 'cadence' in info:
                self.cadences.append(info['cadence'])

        self.logger.record("curriculum/velocity_weight", new_vel_weight)
        self.logger.record("curriculum/tracking_weight", new_track_weight)
        
        # 주기적 로깅 (1000 스텝마다 평균값 기록)
        if len(self.tracking_errors) > 0 and self.n_calls % 1000 == 0:
            self.logger.record("train/tracking_error_mean", np.mean(self.tracking_errors))
            
            # [추가] 보행 분석 그래프 기록
            if len(self.stride_lengths) > 0:
                self.logger.record("gait_analysis/stride_length", np.mean(self.stride_lengths))
                self.logger.record("gait_analysis/stride_time", np.mean(self.stride_times))
                self.logger.record("gait_analysis/cadence", np.mean(self.cadences))
            
            # 버퍼 초기화
            self.tracking_errors = []
            self.stride_lengths = []
            self.stride_times = []
            self.cadences = []
            
        return True

# ==================== 6. 메인 실행 루프 ====================
def make_env(rank, seed=0):
    def _init():
        env = gym.make(Config.env_name)
        env = PhaseAugmentedWrapper(env, Config())
        # [수정] Monitor 추가 (경고 해결 및 원본 점수 로깅)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    print(f" 학습 시작: {Config.env_name}")
    print(f"병렬 환경: {Config.n_envs}개")
    print(f" DUP Mode: {'ON' if Config.enable_mirror_augmentation else 'OFF'}")
    print(f" Gait Analysis: Stride, Cadence Logging ON")
    
    # 1. 학습 환경
    train_env = SubprocVecEnv([make_env(i) for i in range(Config.n_envs)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. 평가 환경
    eval_env = DummyVecEnv([make_env(999)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
    
    # 3. 콜백 설정
    save_vec_norm_cb = SaveVecNormalizeCallback(train_env, os.path.join(Config.log_dir, "vec_normalize.pkl"))
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(Config.log_dir, "best_model"),
        log_path=os.path.join(Config.log_dir, "eval_logs"),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=save_vec_norm_cb,
        verbose=1
    )
    
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
    print(" 학습 및 저장 완료!")
    
    train_env.close()
    eval_env.close()

    # ==================== 7. 테스트 (시각화) ====================
    print("\n 최종 모델 테스트 시작...")
    
    test_env = DummyVecEnv([lambda: PhaseAugmentedWrapper(gym.make(Config.env_name, render_mode="human"), Config())])
    
    stats_path = os.path.join(Config.log_dir, "vec_normalize.pkl")
    if os.path.exists(stats_path):
        test_env = VecNormalize.load(stats_path, test_env)
        test_env.training = False
        test_env.norm_reward = False
        print(" VecNormalize 통계 로드 완료")
    
    model_path = os.path.join(Config.log_dir, "best_model", "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(Config.log_dir, "final_model.zip")
        
    model = SAC.load(model_path)
    
    obs = test_env.reset()
    for _ in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
