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
    total_timesteps = 2_000_000
    
    # 병렬 처리 설정
    n_envs = 8
    
    # 위상 및 궤적 파라미터 (적응형)
    stride_length_base = 0.35   # 기본 보폭 (m)
    stride_length_coef = 0.25   # 속도 계수
    stride_time_base = 0.8      # 기본 주기 (s)
    stride_time_coef = -0.15    # 속도 계수 (음수: 빠르면 짧아짐)
    x_vel_min = 0.1             # 최소 속도 (위상 멈춤 방지)
    
    hip_amplitude = 0.25
    hip_center = 0.0
    hip_offset = 0.0
    knee_amplitude = 0.5
    knee_center = 0.8
    knee_offset = -np.pi/2
    
    # 보상 가중치
    lambda_vel_init = 0.05      
    lambda_vel_final = 1.0       
    lambda_track_init = 0.0     
    lambda_track_final = 1.0   
    lambda_ctrl = 0.0005
    
    # 커리큘럼 설정
    curriculum_vel_steps = 800_000 
    curriculum_track_steps = 1_600_000
    
    # DUP (Mirror Augmentation) 설정
    enable_mirror_augmentation = True
    mirror_probability = 0.5  # 50% 확률로 mirrored 데이터 사용
    
    # 경로 설정
    log_dir = "./walker_phase_sac_logs/"

os.makedirs(Config.log_dir, exist_ok=True)

# ==================== 2. Mirror 함수 (DUP 구현) ====================
class MirrorAugmentation:
    """
    Walker2D-v5의 좌우 대칭 변환
    
    관측 구조 (17차원, exclude_current_positions_from_observation=True):
    - qpos[1:8]: [rootz, rooty, thigh_joint(L), leg_joint(L), foot_joint(L), 
                   thigh_right(R), leg_right(R), foot_right(R)]
    - qvel[0:9]: [rootx_vel, rootz_vel, rooty_vel, thigh_L_vel, leg_L_vel, foot_L_vel,
                   thigh_R_vel, leg_R_vel, foot_R_vel]
    
    행동 구조 (6차원):
    - [thigh_L_torque, leg_L_torque, foot_L_torque,
       thigh_R_torque, leg_R_torque, foot_R_torque]
    """
    
    @staticmethod
    def mirror_obs(obs):
        """
        Walker2D 관측을 좌우 반전
        
        obs 구조 (17차원):
        [0]: rootz (높이) - 유지
        [1]: rooty (각도) - 부호 반전
        [2-4]: 왼쪽 다리 관절 (thigh, leg, foot)
        [5-7]: 오른쪽 다리 관절 (thigh, leg, foot)
        [8]: rootx_vel - 유지 (전방 속도)
        [9]: rootz_vel - 유지
        [10]: rooty_vel - 부호 반전
        [11-13]: 왼쪽 다리 각속도
        [14-16]: 오른쪽 다리 각속도
        """
        mirrored = obs.copy()
        
        # qpos 부분
        mirrored[1] = -obs[1]  # rooty 각도 반전
        
        # 왼쪽 <-> 오른쪽 관절 각도 swap
        mirrored[2:5] = obs[5:8]  # left <- right
        mirrored[5:8] = obs[2:5]  # right <- left
        
        # qvel 부분
        mirrored[10] = -obs[10]  # rooty_vel 반전
        
        # 왼쪽 <-> 오른쪽 각속도 swap
        mirrored[11:14] = obs[14:17]  # left_vel <- right_vel
        mirrored[14:17] = obs[11:14]  # right_vel <- left_vel
        
        return mirrored
    
    @staticmethod
    def mirror_action(action):
        """
        Walker2D 행동을 좌우 반전
        
        action 구조 (6차원):
        [0-2]: 왼쪽 다리 토크 (thigh, leg, foot)
        [3-5]: 오른쪽 다리 토크 (thigh, leg, foot)
        """
        mirrored = action.copy()
        
        # 왼쪽 <-> 오른쪽 토크 swap
        mirrored[0:3] = action[3:6]  # left <- right
        mirrored[3:6] = action[0:3]  # right <- left
        
        return mirrored
    
    @staticmethod
    def mirror_obs_augmented(aug_obs):
        """
        PhaseAugmentedWrapper의 확장된 관측 반전
        
        구조: [base_obs(17), sin(φ)(1), cos(φ)(1), tracking_errors(4)]
        - base_obs: 위와 동일하게 반전
        - phase: 유지 (sin, cos)
        - tracking_errors: 좌우 swap
        """
        mirrored = aug_obs.copy()
        
        # Base observation 반전
        mirrored[:17] = MirrorAugmentation.mirror_obs(aug_obs[:17])
        
        # Phase encoding (sin, cos) 유지
        # mirrored[17:19] = aug_obs[17:19]  # 이미 복사됨
        
        # Tracking errors swap (left <-> right)
        # [left_hip_error, left_knee_error, right_hip_error, right_knee_error]
        mirrored[19:21] = aug_obs[21:23]  # left <- right
        mirrored[21:23] = aug_obs[19:21]  # right <- left
        
        return mirrored

# ==================== 3. 목표 궤적 생성기 ====================
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

# ==================== 4. 환경 래퍼 (DUP 통합) ====================
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
        self.current_track_weight = self.cfg.lambda_track_init
        
        # DUP: Mirror 적용 여부 (에피소드마다 결정)
        self.use_mirror = False
        
        # 적응형 보행 파라미터 추적 (디버깅용)
        self.current_stride_length = 0.0
        self.current_stride_time = 0.0
        
        # Observation Space 확장 (기존 17 + sin + cos + 4 errors = 23)
        base_dim = self.env.observation_space.shape[0]
        new_dim = base_dim + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32
        )
    
    def _get_adaptive_stride_params(self, velocity):
        """
        속도에 따른 보폭과 주기 조정
        
        생체역학 경험식:
        - stride_length ≈ 0.35 + 0.25 * velocity  (m)
        - stride_time ≈ 0.8 - 0.15 * velocity     (s)
        
        예시:
        - v=0.5 m/s: L=0.48m, T=0.73s (느린 걸음)
        - v=1.0 m/s: L=0.60m, T=0.65s (보통 걸음)
        - v=1.5 m/s: L=0.73m, T=0.58s (빠른 걸음)
        """
        v = np.clip(abs(velocity), self.cfg.x_vel_min, 2.0)
        
        stride_length = self.cfg.stride_length_base + self.cfg.stride_length_coef * v
        stride_time = self.cfg.stride_time_base + self.cfg.stride_time_coef * v
        
        # 주기 시간 하한 (너무 빠른 회전 방지)
        stride_time = max(stride_time, 0.4)
        
        return stride_length, stride_time
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.phase = 0.0
        self.time = 0.0
        
        # DUP: 에피소드 시작 시 mirror 여부 결정
        if self.cfg.enable_mirror_augmentation:
            self.use_mirror = np.random.rand() < self.cfg.mirror_probability
        else:
            self.use_mirror = False
        
        aug_obs = self._augment_observation(obs)
        
        # Mirror 적용
        if self.use_mirror:
            aug_obs = MirrorAugmentation.mirror_obs_augmented(aug_obs)
        
        return aug_obs, info
        
    def step(self, action):
        # DUP: Mirror된 에피소드에서는 행동도 반전
        actual_action = action
        if self.use_mirror:
            actual_action = MirrorAugmentation.mirror_action(action)
        
        obs, reward, terminated, truncated, info = self.env.step(actual_action)
        x_vel = info.get('x_velocity', self.env.unwrapped.data.qvel[0])

        # 적응형 보행 파라미터 계산
        stride_length, stride_time = self._get_adaptive_stride_params(x_vel)
        self.current_stride_length = stride_length
        self.current_stride_time = stride_time

        # 위상 업데이트 (시간 기반, 속도 적응형)
        self.time += self.dt
        delta_phi = self.dt / stride_time
        self.phase = (self.phase + delta_phi) % 1.0
        
        aug_obs = self._augment_observation(obs)
        custom_reward, tracking_error_L1 = self._compute_reward(actual_action, info)
        
        # DUP: Mirror된 관측 반환
        if self.use_mirror:
            aug_obs = MirrorAugmentation.mirror_obs_augmented(aug_obs)
        
        # 모니터링 (적응형 파라미터 추가)
        info['tracking_error_L1'] = tracking_error_L1
        info['phase'] = self.phase
        info['vel_weight'] = self.current_vel_weight
        info['is_mirrored'] = self.use_mirror
        info['stride_length'] = stride_length
        info['stride_time'] = stride_time
        info['cadence'] = 60.0 / stride_time  # steps/min
        
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

# ==================== 5. 콜백들 ====================

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, vec_normalize_env, save_path):
        super().__init__()
        self.vec_normalize_env = vec_normalize_env
        self.save_path = save_path
    
    def _on_step(self):
        self.vec_normalize_env.save(self.save_path)
        if self.verbose > 0:
            print(f"VecNormalize 통계 저장 완료: {self.save_path}")
        return True

class CurriculumAndMonitorCallback(BaseCallback):
    """커리큘럼 + DUP + 적응형 보행 모니터링"""
    def __init__(self, config, verbose=0):
        super().__init__(verbose)
        self.cfg = config
        self.tracking_errors = []
        self.mirror_episodes = []
        self.stride_lengths = []
        self.stride_times = []
        self.cadences = []
    
    def _on_step(self) -> bool:
        # 1. 커리큘럼 업데이트
        progress_vel = min(self.num_timesteps / self.cfg.curriculum_vel_steps, 1.0)
        new_vel_weight = self.cfg.lambda_vel_init + progress_vel * (self.cfg.lambda_vel_final - self.cfg.lambda_vel_init)
        
        progress_track = max(0.0, (self.num_timesteps - self.cfg.curriculum_vel_steps) / (self.cfg.curriculum_track_steps - self.cfg.curriculum_vel_steps))
        progress_track = min(progress_track, 1.0)

        new_track_weight = self.cfg.lambda_track_init + progress_track * (self.cfg.lambda_track_final - self.cfg.lambda_track_init)

        self.training_env.env_method("set_velocity_weight", new_vel_weight)
        self.training_env.env_method("set_tracking_weight", new_track_weight)
        
        # 2. 모니터링 (DUP + 적응형 보행)
        infos = self.locals.get("infos", [])
        for info in infos:
            if 'tracking_error_L1' in info:
                self.tracking_errors.append(info['tracking_error_L1'])
            if 'is_mirrored' in info:
                self.mirror_episodes.append(1 if info['is_mirrored'] else 0)
            if 'stride_length' in info:
                self.stride_lengths.append(info['stride_length'])
            if 'stride_time' in info:
                self.stride_times.append(info['stride_time'])
            if 'cadence' in info:
                self.cadences.append(info['cadence'])
                
        # Tensorboard 기록
        self.logger.record("curriculum/velocity_weight", new_vel_weight)
        self.logger.record("curriculum/tracking_weight", new_track_weight)
        
        if len(self.tracking_errors) > 0 and self.n_calls % 1000 == 0:
            mean_error = np.mean(self.tracking_errors)
            self.logger.record("train/tracking_error_mean", mean_error)
            self.tracking_errors = []
        
        if len(self.mirror_episodes) > 0 and self.n_calls % 1000 == 0:
            mirror_ratio = np.mean(self.mirror_episodes)
            self.logger.record("train/mirror_episode_ratio", mirror_ratio)
            self.mirror_episodes = []
        
        # 적응형 보행 메트릭
        if len(self.stride_lengths) > 0 and self.n_calls % 1000 == 0:
            self.logger.record("gait/stride_length_mean", np.mean(self.stride_lengths))
            self.logger.record("gait/stride_time_mean", np.mean(self.stride_times))
            self.logger.record("gait/cadence_mean", np.mean(self.cadences))
            self.stride_lengths = []
            self.stride_times = []
            self.cadences = []
            
        return True

# ==================== 6. 메인 실행 루프 ====================
def make_env(rank, seed=0):
    def _init():
        env = gym.make(Config.env_name)
        env = PhaseAugmentedWrapper(env, Config())
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    print(f"🚀 학습 시작: {Config.env_name}")
    print(f"병렬 환경: {Config.n_envs}개")
    print(f"✨ DUP (Mirror Augmentation): {'활성화' if Config.enable_mirror_augmentation else '비활성화'}")
    print(f"   - Mirror 확률: {Config.mirror_probability * 100:.0f}%")
    print(f"🦵 적응형 보행 파라미터:")
    print(f"   - 보폭: {Config.stride_length_base}m + {Config.stride_length_coef} × v")
    print(f"   - 주기: {Config.stride_time_base}s + {Config.stride_time_coef} × v")
    print(f"   - 예상 범위: 0.48m-0.73m, 0.58s-0.73s\n")
    
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
        eval_freq=2000,
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
    else:
        print("  VecNormalize 통계 없음 - 성능이 저하될 수 있습니다")

    model_path = os.path.join(Config.log_dir, "best_model", "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(Config.log_dir, "final_model.zip")
        
    model = SAC.load(model_path)
    
    obs = test_env.reset()
    for _ in range(2000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        
        if done[0]:
            obs = test_env.reset()

    test_env.close()
    print(" 테스트 완료!")