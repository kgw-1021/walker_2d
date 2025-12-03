import gymnasium as gym
import numpy as np
import torch
import os
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
from datetime import datetime

os.environ['MUJOCO_GL'] = 'glfw'
os.environ['SDL_VIDEODRIVER'] = 'windows'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

# ==================== Config & Wrappers (동일) ====================
class Config:
    env_name = "Walker2d-v5"
    
    # 적응형 보행 파라미터
    stride_length_base = 0.35
    stride_length_coef = 0.25
    stride_time_base = 0.8
    stride_time_coef = -0.15
    x_vel_min = 0.1
    
    hip_amplitude = 0.25
    hip_center = 0.0
    hip_offset = 0.0
    knee_amplitude = 0.5
    knee_center = 0.8
    knee_offset = -np.pi/2
    
    lambda_vel_init = 0.05      
    lambda_vel_final = 1.0       
    lambda_track_init = 0.0     
    lambda_track_final = 1.0   
    lambda_ctrl = 0.0005
    
    curriculum_vel_steps = 600_000  
    curriculum_track_steps = 1_200_000 
    
    log_dir = "./walker_phase_sac_logs/"
    
    # 평가 설정
    n_eval_episodes = 10
    max_test_steps = 1000
    
    # 비디오 설정
    save_video = False
    video_folder = "./walker_phase_sac_logs/videos/"
    video_name_prefix = "walker2d_evaluation"

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

class PhaseAugmentedWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.cfg = config
        self.traj_gen = TrajectoryGenerator(config)
        self.joint_indices = {'left_hip': 3, 'left_knee': 4, 'right_hip': 6, 'right_knee': 7}
        
        self.phase = 0.0
        self.time = 0.0
        self.dt = 0.008
        self.current_vel_weight = self.cfg.lambda_vel_final
        self.current_track_weight = self.cfg.lambda_track_final
        
        # 평가용 추가 변수
        self.episode_energy = 0.0
        self.episode_distance = 0.0
        self.initial_x = 0.0
        
        base_dim = self.env.observation_space.shape[0]
        new_dim = base_dim + 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32
        )
    
    def _get_adaptive_stride_params(self, velocity):
        v = np.clip(abs(velocity), self.cfg.x_vel_min, 2.0)
        stride_length = self.cfg.stride_length_base + self.cfg.stride_length_coef * v
        stride_time = self.cfg.stride_time_base + self.cfg.stride_time_coef * v
        stride_time = max(stride_time, 0.4)
        return stride_length, stride_time
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.phase = 0.0
        self.time = 0.0
        self.episode_energy = 0.0
        self.episode_distance = 0.0
        self.initial_x = self.env.unwrapped.data.qpos[0]
        return self._augment_observation(obs), info
        
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        x_vel = info.get('x_velocity', self.env.unwrapped.data.qvel[0])
        
        # 적응형 파라미터
        stride_length, stride_time = self._get_adaptive_stride_params(x_vel)
        
        self.time += self.dt
        delta_phi = self.dt / stride_time
        self.phase = (self.phase + delta_phi) % 1.0
        
        aug_obs = self._augment_observation(obs)
        custom_reward, tracking_error_L1 = self._compute_reward(action, info)
        
        # ===== 평가 메트릭 계산 =====
        qpos = self.env.unwrapped.data.qpos
        qvel = self.env.unwrapped.data.qvel
        
        # 1. 에너지 소모 (CoT 계산용)
        energy_step = np.sum(np.abs(action * qvel[3:9]))  # torque * angular_velocity
        self.episode_energy += energy_step * self.dt
        
        # 2. 이동 거리
        current_x = qpos[0]
        self.episode_distance = current_x - self.initial_x
        
        # 3. 대칭성 오차 (위상 보정 - 교차 보행)
        left_hip = qpos[3]
        left_knee = qpos[4]
        right_hip = qpos[6]
        right_knee = qpos[7]
        
        # 목표 궤적 가져오기 (왼쪽/오른쪽 각각의 목표)
        targets = self.traj_gen.get_target_angles(self.phase)
        
        # 각 다리의 궤적 추종 오차 계산
        left_error = abs(left_hip - targets['left_hip']) + abs(left_knee - targets['left_knee'])
        right_error = abs(right_hip - targets['right_hip']) + abs(right_knee - targets['right_knee'])
        
        # 대칭성: 좌우가 각자의 목표를 얼마나 비슷하게 추종하는지
        # (왼쪽 오차와 오른쪽 오차의 차이가 작을수록 대칭)
        symmetry_error = abs(left_error - right_error)
        
        # 4. 몸통 안정성 (rooty 각도)
        torso_angle = abs(qpos[2])  # rooty
        
        # Info에 메트릭 추가
        info['energy_step'] = energy_step
        info['total_energy'] = self.episode_energy
        info['distance'] = self.episode_distance
        info['symmetry_error'] = symmetry_error
        info['torso_angle'] = torso_angle
        info['x_velocity'] = x_vel
        info['tracking_error_L1'] = tracking_error_L1
        info['stride_length'] = stride_length
        info['stride_time'] = stride_time
        
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

def make_eval_env(config, render_mode=None, save_video=False):
    env = gym.make(config.env_name, render_mode=render_mode)
    env = PhaseAugmentedWrapper(env, config)
    
    # 비디오 저장 래퍼 추가
    if save_video:
        os.makedirs(config.video_folder, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=config.video_folder,
            episode_trigger=lambda x: True,  # 모든 에피소드 녹화
            name_prefix=config.video_name_prefix
        )
        print(f"✓ 비디오 저장 설정: {config.video_folder}")
    
    return Monitor(env)

# ==================== 개선된 평가 함수 ====================

def evaluate_with_advanced_metrics(config, render=False, save_video=False):
    """개선된 메트릭으로 평가"""
    
    # 1. 환경 및 모델 로드
    stats_path = os.path.join(config.log_dir, "vec_normalize.pkl")
    model_path = os.path.join(config.log_dir, "best_model", "best_model.zip")
    
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print(" 오류: 모델 또는 통계 파일 없음")
        return

    # 비디오 저장 시에는 rgb_array 모드 필수
    render_mode = "rgb_array" if save_video else ("human" if render else None)
    
    # 환경 생성 (비디오 저장 옵션 포함)
    eval_env = DummyVecEnv([lambda: make_eval_env(config, render_mode=render_mode, save_video=save_video)])
    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    
    model = SAC.load(model_path)
    print(f"✅ 모델 로드: {model_path}")
    
    if save_video:
        print(f"🎥 비디오 녹화 모드 활성화\n")

    # 2. 평가 메트릭 저장
    metrics = {
        'rewards': [],
        'velocities': [],
        'distances': [],
        'energies': [],
        'cots': [],  # Cost of Transport
        'symmetry_errors': [],
        'torso_angles': [],
        'survival_rates': [],
        'stride_lengths': [],
        'stride_times': []
    }
    
    print(f"📊 {config.n_eval_episodes}개 에피소드 평가 시작...\n")
    
    for episode in range(config.n_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        # 에피소드별 메트릭
        ep_velocities = []
        ep_symmetry_errors = []
        ep_torso_angles = []
        ep_stride_lengths = []
        ep_stride_times = []
        
        while not done and step_count < config.max_test_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            episode_reward += reward[0]
            step_count += 1
            
            info_data = info[0]
            
            # 실시간 메트릭 수집
            if 'x_velocity' in info_data:
                ep_velocities.append(info_data['x_velocity'])
            if 'symmetry_error' in info_data:
                ep_symmetry_errors.append(info_data['symmetry_error'])
            if 'torso_angle' in info_data:
                ep_torso_angles.append(info_data['torso_angle'])
            if 'stride_length' in info_data:
                ep_stride_lengths.append(info_data['stride_length'])
            if 'stride_time' in info_data:
                ep_stride_times.append(info_data['stride_time'])
                
            done = done[0]

        # 에피소드 종료 후 메트릭 계산
        if step_count > 0:
            final_info = info[0]
            
            # 기본 메트릭
            avg_velocity = np.mean(ep_velocities) if ep_velocities else 0
            distance = final_info.get('distance', 0)
            total_energy = final_info.get('total_energy', 0)
            
            # Cost of Transport (낮을수록 효율적)
            # CoT = E / (m * d), Walker2d mass ≈ 3.5kg (추정)
            mass = 3.5
            cot = total_energy / (mass * distance) if distance > 0 else float('inf')
            
            # 대칭성 (낮을수록 좋음)
            avg_symmetry = np.mean(ep_symmetry_errors) if ep_symmetry_errors else 0
            
            # 안정성 (몸통 각도, 낮을수록 안정)
            avg_torso_angle = np.mean(ep_torso_angles) if ep_torso_angles else 0
            
            # 생존율 (넘어지지 않고 유지)
            survival_rate = step_count / config.max_test_steps
            
            # 보행 파라미터
            avg_stride_length = np.mean(ep_stride_lengths) if ep_stride_lengths else 0
            avg_stride_time = np.mean(ep_stride_times) if ep_stride_times else 0
            
            # 저장
            metrics['rewards'].append(episode_reward)
            metrics['velocities'].append(avg_velocity)
            metrics['distances'].append(distance)
            metrics['energies'].append(total_energy)
            metrics['cots'].append(cot)
            metrics['symmetry_errors'].append(avg_symmetry)
            metrics['torso_angles'].append(avg_torso_angle)
            metrics['survival_rates'].append(survival_rate)
            metrics['stride_lengths'].append(avg_stride_length)
            metrics['stride_times'].append(avg_stride_time)
            
            # 에피소드별 출력
            print(f"[EP {episode+1:2d}] "
                  f"보상: {episode_reward:6.1f} | "
                  f"속도: {avg_velocity:.2f}m/s | "
                  f"거리: {distance:5.1f}m | "
                  f"대칭: {avg_symmetry:.3f} | "
                  f"생존: {survival_rate:.1%}")

    eval_env.close()
    
    # 비디오 저장 완료 메시지
    if save_video:
        print(f"\n🎥 비디오 저장 완료: {config.video_folder}")
        print(f"   - 파일명 형식: {config.video_name_prefix}-episode-*.mp4")
    
    # 3. 최종 통계 출력
    print("\n" + "="*70)
    print("                      🏆 최종 평가 결과")
    print("="*70)
    
    print(f"\n📈 성능 메트릭 ({config.n_eval_episodes}개 에피소드 평균)")
    print("-"*70)
    print(f"  1. 총 보상 (Total Reward):           {np.mean(metrics['rewards']):8.2f} ± {np.std(metrics['rewards']):.2f}")
    print(f"  2. 평균 속도 (Avg Velocity):         {np.mean(metrics['velocities']):8.2f} m/s ± {np.std(metrics['velocities']):.2f}")
    print(f"  3. 총 이동 거리 (Distance):          {np.mean(metrics['distances']):8.2f} m ± {np.std(metrics['distances']):.2f}")
    print(f"  4. 생존율 (Survival Rate):           {np.mean(metrics['survival_rates']):8.1%} ± {np.std(metrics['survival_rates']):.1%}")
    
    print(f"\n⚡ 효율성 메트릭")
    print("-"*70)
    valid_cots = [c for c in metrics['cots'] if c != float('inf')]
    if valid_cots:
        print(f"  5. CoT (Cost of Transport):          {np.mean(valid_cots):8.3f} ± {np.std(valid_cots):.3f}")
        print(f"     (에너지/거리, 낮을수록 효율적)")
    print(f"  6. 총 에너지 소모:                   {np.mean(metrics['energies']):8.2f} J ± {np.std(metrics['energies']):.2f}")
    
    print(f"\n🎯 보행 품질 메트릭")
    print("-"*70)
    print(f"  7. 대칭성 오차 (Symmetry Error):     {np.mean(metrics['symmetry_errors']):8.4f} rad ± {np.std(metrics['symmetry_errors']):.4f}")
    print(f"     (좌우 다리의 궤적 추종 오차 차이, 0에 가까울수록 대칭)")
    print(f"     (위상 180도 차이를 고려한 교차 보행 대칭성)")
    print(f"  8. 몸통 안정성 (Torso Angle):        {np.mean(metrics['torso_angles']):8.4f} rad ± {np.std(metrics['torso_angles']):.4f}")
    print(f"     (몸통 기울기, 0에 가까울수록 안정)")
    
    print(f"\n🦵 보행 파라미터")
    print("-"*70)
    print(f"  9. 평균 보폭 (Stride Length):        {np.mean(metrics['stride_lengths']):8.3f} m ± {np.std(metrics['stride_lengths']):.3f}")
    print(f" 10. 평균 주기 (Stride Time):          {np.mean(metrics['stride_times']):8.3f} s ± {np.std(metrics['stride_times']):.3f}")
    avg_cadence = 60 / np.mean(metrics['stride_times']) if np.mean(metrics['stride_times']) > 0 else 0
    print(f" 11. 케이던스 (Cadence):               {avg_cadence:8.1f} steps/min")
    
    # 4. 종합 점수 계산
    print(f"\n🌟 종합 보행 품질 점수 (Gait Quality Score)")
    print("-"*70)
    
    # 정규화된 점수 (0-100)
    velocity_score = min(np.mean(metrics['velocities']) / 1.5 * 100, 100)  # 1.5 m/s = 100점
    symmetry_score = max(0, 100 - np.mean(metrics['symmetry_errors']) * 100)  # 낮을수록 좋음
    stability_score = max(0, 100 - np.mean(metrics['torso_angles']) * 200)  # 낮을수록 좋음
    survival_score = np.mean(metrics['survival_rates']) * 100
    
    if valid_cots:
        efficiency_score = max(0, 100 - np.mean(valid_cots) * 10)  # 낮을수록 좋음
    else:
        efficiency_score = 0
    
    overall_score = (
        velocity_score * 0.25 + 
        symmetry_score * 0.25 + 
        stability_score * 0.2 + 
        efficiency_score * 0.15 +
        survival_score * 0.15
    )
    
    print(f"  - 속도 점수:      {velocity_score:5.1f}/100")
    print(f"  - 대칭 점수:      {symmetry_score:5.1f}/100")
    print(f"  - 안정성 점수:    {stability_score:5.1f}/100")
    print(f"  - 효율성 점수:    {efficiency_score:5.1f}/100")
    print(f"  - 생존율 점수:    {survival_score:5.1f}/100")
    print(f"\n  ⭐ 최종 점수:     {overall_score:5.1f}/100")
    
    print("="*70)
    
    # 5. 시각화 (선택)
    plot_metrics(metrics, config)
    
    return metrics

def plot_metrics(metrics, config):
    """메트릭 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Walker2D 평가 메트릭', fontsize=16, fontweight='bold')
    
    # 1. 속도
    axes[0, 0].bar(range(len(metrics['velocities'])), metrics['velocities'], color='skyblue')
    axes[0, 0].axhline(np.mean(metrics['velocities']), color='red', linestyle='--', label='평균')
    axes[0, 0].set_title('평균 속도 (m/s)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].legend()
    
    # 2. 대칭성
    axes[0, 1].bar(range(len(metrics['symmetry_errors'])), metrics['symmetry_errors'], color='lightcoral')
    axes[0, 1].axhline(np.mean(metrics['symmetry_errors']), color='red', linestyle='--', label='평균')
    axes[0, 1].set_title('대칭성 오차 (rad, 낮을수록 좋음)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].legend()
    
    # 3. CoT
    valid_cots = [c if c != float('inf') else 0 for c in metrics['cots']]
    axes[0, 2].bar(range(len(valid_cots)), valid_cots, color='lightgreen')
    axes[0, 2].set_title('Cost of Transport (낮을수록 효율적)')
    axes[0, 2].set_xlabel('Episode')
    
    # 4. 보폭
    axes[1, 0].bar(range(len(metrics['stride_lengths'])), metrics['stride_lengths'], color='orange')
    axes[1, 0].axhline(np.mean(metrics['stride_lengths']), color='red', linestyle='--')
    axes[1, 0].set_title('평균 보폭 (m)')
    axes[1, 0].set_xlabel('Episode')
    
    # 5. 안정성
    axes[1, 1].bar(range(len(metrics['torso_angles'])), metrics['torso_angles'], color='plum')
    axes[1, 1].axhline(np.mean(metrics['torso_angles']), color='red', linestyle='--')
    axes[1, 1].set_title('몸통 각도 (rad, 0에 가까울수록 안정)')
    axes[1, 1].set_xlabel('Episode')
    
    # 6. 생존율
    axes[1, 2].bar(range(len(metrics['survival_rates'])), metrics['survival_rates'], color='mediumpurple')
    axes[1, 2].axhline(np.mean(metrics['survival_rates']), color='red', linestyle='--')
    axes[1, 2].set_title('생존율 (높을수록 안정)')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylim([0, 1.1])
    
    plt.tight_layout()
    save_path = os.path.join(config.log_dir, 'evaluation_metrics.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 그래프 저장: {save_path}")
    plt.show()

if __name__ == "__main__":
    config = Config()
    
    # 비디오 저장 여부 선택
    save_video = config.save_video  # Config에서 설정
    
    metrics = evaluate_with_advanced_metrics(config, render=False, save_video=save_video)