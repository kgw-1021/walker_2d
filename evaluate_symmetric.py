import gymnasium as gym
import numpy as np
import torch
import os
from gymnasium import spaces

# Stable Baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt

# ==================== A. 환경 및 설정 재정의 (learn_symmetric.py와 동일) ====================
# 학습 코드에서 사용된 Config, TrajectoryGenerator, PhaseAugmentedWrapper 클래스를 그대로 사용합니다.

os.environ['MUJOCO_GL'] = 'glfw'
os.environ['SDL_VIDEODRIVER'] = 'windows'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

class Config:
    # 학습 시와 동일한 환경 및 경로 설정 사용
    env_name = "Walker2d-v5"
    
    # 위상 및 궤적 파라미터 (학습 시와 동일해야 함)
    T_stride = 0.7
    hip_amplitude = 0.25
    hip_center = 0.0
    hip_offset = 0.0
    knee_amplitude = 0.5
    knee_center = 0.8
    knee_offset = -np.pi/2
    
    # 보상 가중치 (평가 시 사용되지는 않지만 Wrapper 초기화에 필요)
    lambda_vel_init = 0.05      
    lambda_vel_final = 1.0       
    lambda_track_init = 1.0     # 평가 시에는 1.0으로 가정 (최종 가중치)
    lambda_track_final = 1.0   
    lambda_ctrl = 0.0005
    
    # 커리큘럼 설정 (평가 시 사용되지 않음)
    curriculum_vel_steps = 700_000 
    curriculum_track_steps = 1_400_000 
    
    # 경로 설정 (학습 코드의 로그 경로와 일치해야 함)
    log_dir = "./walker_phase_sac_logs/"
    
    # 평가 설정
    n_eval_episodes = 10
    max_test_steps = 1000

class TrajectoryGenerator:
    # 학습 코드와 동일 (생략)
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

class PhaseAugmentedWrapper(gym.Wrapper):
    # 학습 코드와 동일 (L1 Norm, Observation 확장 포함)
    def __init__(self, env, config):
        super().__init__(env)
        self.cfg = config
        self.traj_gen = TrajectoryGenerator(config)
        self.joint_indices = {'left_hip': 3, 'left_knee': 4, 'right_hip': 6, 'right_knee': 7}
        
        self.phase = 0.0
        self.time = 0.0
        self.dt = 0.008
        
        # 평가 시에는 최종 가중치로 고정하여 사용
        self.current_vel_weight = self.cfg.lambda_vel_final
        self.current_track_weight = self.cfg.lambda_track_final 
        
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
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        
        self.time += self.dt
        self.phase = (self.time / self.cfg.T_stride) % 1.0
        
        aug_obs = self._augment_observation(obs)
        custom_reward, tracking_error_L1 = self._compute_reward(action, info)
        
        info['tracking_error_L1'] = tracking_error_L1 # ACTE 계산을 위해 L1 오차를 info에 기록
        
        # 평가 루프에서 필요한 데이터도 info에 저장
        info['x_velocity'] = info.get('x_velocity', self.env.unwrapped.data.qvel[0])
        
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
        healthy_reward = 5.0 # 학습 시와 동일
        reward = (
            self.current_vel_weight * x_vel
            + healthy_reward
            - self.current_track_weight * tracking_error_L1
            - self.cfg.lambda_ctrl * control_cost
        )
        return reward, tracking_error_L1

# 환경 생성 헬퍼 함수
def make_eval_env(config, render_mode=None):
    env = gym.make(config.env_name, render_mode=render_mode)
    env = PhaseAugmentedWrapper(env, config)
    return Monitor(env)

# ==================== B. 평가 메인 함수 ====================

def evaluate_and_report(config):
    # 1. 모델 및 통계 경로 설정
    stats_path = os.path.join(config.log_dir, "vec_normalize.pkl")
    model_path = os.path.join(config.log_dir, "best_model", "best_model.zip")
    
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print(" 오류: 학습된 모델 파일 또는 VecNormalize 통계 파일을 찾을 수 없습니다.")
        print(f"  모델 경로: {model_path}")
        print(f"  통계 경로: {stats_path}")
        return

    # 2. 평가 환경 준비 (렌더링 없이)
    eval_env = DummyVecEnv([lambda: make_eval_env(config, render_mode="human")])
    eval_env = VecNormalize.load(stats_path, eval_env)
    
    # 평가 모드 설정 (필수!)
    eval_env.training = False
    eval_env.norm_reward = False 
    
    # 3. 모델 로드
    model = SAC.load(model_path)
    print(f"모델 로드 완료: {model_path}")

    # 4. 평가 루프 실행
    all_rewards = []
    all_velocities = []
    all_tracking_errors = []
    
    print(f"\n {config.n_eval_episodes}개 에피소드 평가 시작...")
    
    for episode in range(config.n_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        episode_vel_sum = 0
        episode_track_sum = 0
        step_count = 0
        
        while not done and step_count < config.max_test_steps:
            # 결정론적 예측 (deterministic=True)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            episode_reward += reward[0]
            step_count += 1
            
            # 정보 추출 (Wrapper가 info에 기록한 데이터)
            info_data = info[0]
            if 'x_velocity' in info_data:
                episode_vel_sum += info_data['x_velocity']
            if 'tracking_error_L1' in info_data:
                episode_track_sum += info_data['tracking_error_L1']
                
            done = done[0] # DummyVecEnv의 done은 배열이므로 첫 번째 요소 사용

        # 에피소드 종료 후 결과 기록
        if step_count > 0:
            avg_velocity = episode_vel_sum / step_count
            avg_tracking_error = episode_track_sum / step_count
            
            all_rewards.append(episode_reward)
            all_velocities.append(avg_velocity)
            all_tracking_errors.append(avg_tracking_error)
            
            print(f"  [EP {episode+1:2d}] 총 보상: {episode_reward:.2f}, 평균 속도: {avg_velocity:.2f} m/s, ACTE: {avg_tracking_error:.4f}")

    # 5. 최종 결과 보고
    if len(all_rewards) > 0:
        mean_reward = np.mean(all_rewards)
        mean_velocity = np.mean(all_velocities)
        mean_acte = np.mean(all_tracking_errors)
        
        print("\n" + "="*50)
        print("                   최종 평가 요약 ")
        print("="*50)
        print(f"  - 평가 에피소드 수: {config.n_eval_episodes}개")
        print("-" * 50)
        print(f"  1. 총 보상 (평균): {mean_reward:.2f}")
        print(f"  2. 평균 속도 (X-Vel): {mean_velocity:.2f} m/s")
        print(f"  3. **대칭 보행 정확도 (ACTE)**: {mean_acte:.4f} (낮을수록 좋음)")
        print("="*50)

if __name__ == "__main__":
    config = Config()
    evaluate_and_report(config)