import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch
import os
from typing import Optional
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------
# 1. 환경 설정
# ---------------------------------------------
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['SDL_VIDEODRIVER'] = 'windows'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

# ---------------------------------------------
# 2. 보상 분석 래퍼
# ---------------------------------------------
class RewardAnalysisWrapper(gym.Wrapper):
    """
    Walker2d-v5의 보상 구성 요소를 추적하는 래퍼
    """
    def __init__(self, env):
        super().__init__(env)
        self.reward_components = defaultdict(list)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Walker2d-v5 보상 구성 요소 계산
        x_velocity = info.get('x_velocity', obs[8])
        
        # 1. Forward reward (앞으로 가는 보상)
        forward_reward = info.get('reward_forward', x_velocity)
        
        # 2. Control cost (제어 비용)
        ctrl_cost = info.get('reward_ctrl', -0.001 * np.sum(np.square(action)))
        
        # 3. Healthy reward (생존 보상)
        healthy_reward = info.get('reward_survive', 1.0)
        
        # 보상 구성 요소 저장
        self.reward_components['forward_reward'].append(forward_reward)
        self.reward_components['ctrl_cost'].append(ctrl_cost)
        self.reward_components['healthy_reward'].append(healthy_reward)
        self.reward_components['total_reward'].append(reward)
        self.reward_components['x_velocity'].append(x_velocity)
        
        # 추가 정보
        if len(obs) > 0:
            torso_height = obs[0]
            self.reward_components['torso_height'].append(torso_height)
        
        # info에 보상 분석 추가
        info['reward_analysis'] = {
            'forward_reward': forward_reward,
            'ctrl_cost': ctrl_cost,
            'healthy_reward': healthy_reward,
            'total_reward': reward,
            'x_velocity': x_velocity
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        # 새 에피소드 시작 시 버퍼 초기화
        self.reward_components = defaultdict(list)
        return self.env.reset(**kwargs)
    
    def get_reward_statistics(self):
        """현재까지의 보상 통계 반환"""
        stats = {}
        for key, values in self.reward_components.items():
            if len(values) > 0:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'sum': np.sum(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return stats

# ---------------------------------------------
# 3. 장시간 속도 분석 함수 (신규)
# ---------------------------------------------
def analyze_long_run_speed(
    model_path: str = "./walker2d_speed_boost_logs/best_model/best_model.zip",
    vec_normalize_path: str = "./walker2d_speed_boost_logs/vec_normalize_best.pkl",
    total_steps: int = 50000,
    warmup_steps: int = 500,
    save_plots: bool = True,
    plot_folder: str = "./speed_analysis/",
    render: bool = False
):
    """
    50,000 스텝 동안 실행하면서 속도를 추적하고 분석합니다.
    
    Args:
        model_path: 학습된 모델 경로
        vec_normalize_path: VecNormalize 통계 파일 경로
        total_steps: 총 실행 스텝 수
        warmup_steps: 통계 수집 시작 전 워밍업 스텝
        save_plots: 플롯 저장 여부
        plot_folder: 플롯 저장 폴더
        render: 렌더링 여부
    """
    
    print("\n" + "="*70)
    print(f"장시간 속도 분석 시작 ({total_steps:,} 스텝)")
    print("="*70)
    
    # 파일 확인
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 플롯 폴더 생성
    if save_plots:
        os.makedirs(plot_folder, exist_ok=True)
    
    # 환경 생성
    render_mode = "human" if render else None
    env = gym.make("Walker2d-v5", render_mode=render_mode)
    env = Monitor(env)
    
    # 모델 로드
    print(f"✓ 모델 로드: {model_path}")
    model = SAC.load(model_path, device=device)
    
    # 정규화 통계 로드
    obs_mean = None
    obs_std = None
    if os.path.exists(vec_normalize_path):
        temp_env = DummyVecEnv([lambda: gym.make("Walker2d-v5")])
        vec_norm = VecNormalize.load(vec_normalize_path, temp_env)
        obs_mean = vec_norm.obs_rms.mean
        obs_std = np.sqrt(vec_norm.obs_rms.var + 1e-8)
        temp_env.close()
        print("✓ 정규화 통계 로드 완료")
    
    # 데이터 수집 배열
    all_velocities = []
    all_rewards = []
    step_indices = []
    
    # 워밍업 후 속도 데이터
    post_warmup_velocities = []
    
    # 실행
    obs, _ = env.reset()
    current_step = 0
    episode_count = 0
    
    print(f"\n{total_steps:,} 스텝 실행 중...")
    print(f"워밍업 기간: {warmup_steps} 스텝 (이후 통계 수집 시작)")
    
    while current_step < total_steps:
        # 정규화
        if obs_mean is not None:
            obs_norm = np.clip((obs - obs_mean) / obs_std, -10.0, 10.0)
        else:
            obs_norm = obs
        
        # 행동 예측
        action, _ = model.predict(obs_norm, deterministic=True)
        
        # 환경 스텝
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 데이터 수집
        x_velocity = info.get('x_velocity', 0.0)
        all_velocities.append(x_velocity)
        all_rewards.append(reward)
        step_indices.append(current_step)
        
        # 워밍업 이후 데이터 수집
        if current_step >= warmup_steps:
            post_warmup_velocities.append(x_velocity)
        
        current_step += 1
        
        # 진행률 출력
        if current_step % 5000 == 0:
            progress = (current_step / total_steps) * 100
            current_avg_speed = np.mean(all_velocities[-1000:]) if len(all_velocities) >= 1000 else np.mean(all_velocities)
            print(f"  진행률: {progress:.1f}% ({current_step:,}/{total_steps:,} 스텝) - 현재 평균 속도: {current_avg_speed:.2f} m/s")
        
        # 에피소드 종료 시 리셋
        if done:
            episode_count += 1
            obs, _ = env.reset()
    
    env.close()
    
    # ---------------------------------------------
    # 통계 분석
    # ---------------------------------------------
    print("\n" + "="*70)
    print("속도 분석 결과")
    print("="*70)
    
    # 전체 통계
    print(f"\n[전체 구간: 0 ~ {total_steps:,} 스텝]")
    print(f"  평균 속도: {np.mean(all_velocities):.3f} m/s")
    print(f"  표준편차:  {np.std(all_velocities):.3f} m/s")
    print(f"  최고 속도: {np.max(all_velocities):.3f} m/s")
    print(f"  최저 속도: {np.min(all_velocities):.3f} m/s")
    print(f"  중간값:    {np.median(all_velocities):.3f} m/s")
    
    # 워밍업 이후 통계
    print(f"\n[워밍업 이후: {warmup_steps} ~ {total_steps:,} 스텝]")
    print(f"  평균 속도: {np.mean(post_warmup_velocities):.3f} m/s")
    print(f"  표준편차:  {np.std(post_warmup_velocities):.3f} m/s")
    print(f"  최고 속도: {np.max(post_warmup_velocities):.3f} m/s")
    print(f"  최저 속도: {np.min(post_warmup_velocities):.3f} m/s")
    print(f"  중간값:    {np.median(post_warmup_velocities):.3f} m/s")
    
    # 구간별 통계 (10,000 스텝 단위)
    print(f"\n[구간별 평균 속도 (10,000 스텝 단위)]")
    interval = 10000
    for i in range(0, total_steps, interval):
        end_idx = min(i + interval, total_steps)
        interval_velocities = all_velocities[i:end_idx]
        print(f"  {i:6,} ~ {end_idx:6,} 스텝: {np.mean(interval_velocities):.3f} m/s (±{np.std(interval_velocities):.3f})")
    
    print(f"\n총 에피소드 수: {episode_count}")
    
    # ---------------------------------------------
    # 시각화
    # ---------------------------------------------
    if save_plots:
        print(f"\n시각화 저장 중... ({plot_folder})")
        
        # 1. 속도 시계열 플롯
        fig, ax = plt.subplots(figsize=(16, 6))
        
        ax.plot(step_indices, all_velocities, alpha=0.6, linewidth=0.5, color='blue', label='Velocity')
        
        # 워밍업 구간 표시
        ax.axvline(x=warmup_steps, color='red', linestyle='--', linewidth=2, label=f'Warmup End ({warmup_steps} steps)')
        
        # 이동 평균 (1000 스텝)
        window_size = 1000
        if len(all_velocities) >= window_size:
            moving_avg = np.convolve(all_velocities, np.ones(window_size)/window_size, mode='valid')
            moving_avg_indices = step_indices[window_size-1:]
            ax.plot(moving_avg_indices, moving_avg, color='red', linewidth=2, label=f'Moving Average ({window_size} steps)')
        
        # 워밍업 이후 평균선
        post_warmup_mean = np.mean(post_warmup_velocities)
        ax.axhline(y=post_warmup_mean, color='green', linestyle='--', linewidth=2, 
                   label=f'Post-Warmup Mean ({post_warmup_mean:.2f} m/s)')
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Velocity (m/s)', fontsize=12)
        ax.set_title(f'속도 시계열 분석 ({total_steps:,} 스텝)', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, 'velocity_timeseries.png'), dpi=300)
        print(f"  ✓ velocity_timeseries.png 저장 완료")
        plt.close()
        
        # 2. 속도 분포 히스토그램 (전체 vs 워밍업 이후)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(all_velocities, bins=100, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(np.mean(all_velocities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_velocities):.2f}')
        axes[0].set_xlabel('Velocity (m/s)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('전체 속도 분포', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        axes[1].hist(post_warmup_velocities, bins=100, alpha=0.7, color='green', edgecolor='black')
        axes[1].axvline(np.mean(post_warmup_velocities), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(post_warmup_velocities):.2f}')
        axes[1].set_xlabel('Velocity (m/s)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'워밍업 이후 속도 분포 (>{warmup_steps} steps)', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, 'velocity_distribution.png'), dpi=300)
        print(f"  ✓ velocity_distribution.png 저장 완료")
        plt.close()
        
        # 3. 구간별 평균 속도 막대 그래프
        fig, ax = plt.subplots(figsize=(12, 6))
        
        interval = 10000
        intervals = []
        interval_means = []
        interval_stds = []
        
        for i in range(0, total_steps, interval):
            end_idx = min(i + interval, total_steps)
            interval_velocities = all_velocities[i:end_idx]
            intervals.append(f"{i//1000}-{end_idx//1000}k")
            interval_means.append(np.mean(interval_velocities))
            interval_stds.append(np.std(interval_velocities))
        
        x_pos = np.arange(len(intervals))
        bars = ax.bar(x_pos, interval_means, yerr=interval_stds, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 워밍업 구간 강조
        warmup_interval_idx = warmup_steps // interval
        if warmup_interval_idx < len(bars):
            bars[warmup_interval_idx].set_color('orange')
            bars[warmup_interval_idx].set_label('Warmup Interval')
        
        ax.set_xlabel('Step Interval (k = 1000)', fontsize=12)
        ax.set_ylabel('Average Velocity (m/s)', fontsize=12)
        ax.set_title('구간별 평균 속도', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(intervals, rotation=45, ha='right')
        ax.axhline(y=post_warmup_mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Post-Warmup Mean ({post_warmup_mean:.2f} m/s)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, 'interval_velocity.png'), dpi=300)
        print(f"  ✓ interval_velocity.png 저장 완료")
        plt.close()
        
        # 4. 속도 박스플롯 (구간별)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        interval_data = []
        interval_labels = []
        for i in range(0, total_steps, interval):
            end_idx = min(i + interval, total_steps)
            interval_velocities = all_velocities[i:end_idx]
            interval_data.append(interval_velocities)
            interval_labels.append(f"{i//1000}-{end_idx//1000}k")
        
        bp = ax.boxplot(interval_data, labels=interval_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Step Interval (k = 1000)', fontsize=12)
        ax.set_ylabel('Velocity (m/s)', fontsize=12)
        ax.set_title('구간별 속도 분포 (Box Plot)', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_folder, 'velocity_boxplot.png'), dpi=300)
        print(f"  ✓ velocity_boxplot.png 저장 완료")
        plt.close()
        
        print(f"\n모든 플롯이 {plot_folder}에 저장되었습니다.")
    
    print("="*70)
    print("속도 분석 완료!\n")
    
    return {
        'all_velocities': all_velocities,
        'post_warmup_velocities': post_warmup_velocities,
        'all_rewards': all_rewards,
        'step_indices': step_indices,
        'episode_count': episode_count
    }

# ---------------------------------------------
# 4. 보상 분석 함수 (기존)
# ---------------------------------------------
def analyze_rewards(
    model_path: str = "./walker2d_baseline_logs/best_model/best_model.zip",
    vec_normalize_path: str = "./walker2d_baseline_logs/vec_normalize_best.pkl",
    num_episodes: int = 10,
    save_plots: bool = True,
    plot_folder: str = "./reward_analysis/"
):
    """
    학습된 모델의 보상 구성 요소를 상세 분석합니다.
    """
    
    print("\n" + "="*70)
    print("보상 구성 요소 분석 시작")
    print("="*70)
    
    # 파일 확인
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 플롯 폴더 생성
    if save_plots:
        os.makedirs(plot_folder, exist_ok=True)
    
    # 환경 생성 (분석 래퍼 추가)
    env = gym.make("Walker2d-v5")
    env = Monitor(env)
    env = RewardAnalysisWrapper(env)
    
    # 모델 로드
    print(f"✓ 모델 로드: {model_path}")
    model = SAC.load(model_path, device=device)
    
    # 정규화 통계 로드
    obs_mean = None
    obs_std = None
    if os.path.exists(vec_normalize_path):
        temp_env = DummyVecEnv([lambda: gym.make("Walker2d-v5")])
        vec_norm = VecNormalize.load(vec_normalize_path, temp_env)
        obs_mean = vec_norm.obs_rms.mean
        obs_std = np.sqrt(vec_norm.obs_rms.var + 1e-8)
        temp_env.close()
        print("✓ 정규화 통계 로드 완료")
    
    # 전체 에피소드 데이터 수집
    all_episodes_data = []
    episode_summaries = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_data = defaultdict(list)
        done = False
        step_count = 0
        
        print(f"에피소드 {episode + 1}/{num_episodes} 분석 중...", end="\r")
        
        while not done:
            # 정규화
            if obs_mean is not None:
                obs_norm = np.clip((obs - obs_mean) / obs_std, -10.0, 10.0)
            else:
                obs_norm = obs
            
            # 행동 예측
            action, _ = model.predict(obs_norm, deterministic=True)
            
            # 환경 스텝
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 보상 분석 데이터 수집
            if 'reward_analysis' in info:
                for key, value in info['reward_analysis'].items():
                    episode_data[key].append(value)
            
            step_count += 1
        
        # 에피소드 통계 저장
        stats = env.get_reward_statistics()
        episode_summaries.append({
            'episode': episode + 1,
            'steps': step_count,
            'stats': stats
        })
        
        # 전체 데이터에 추가
        all_episodes_data.append(episode_data)
        
        print(f"에피소드 {episode + 1}/{num_episodes} 완료! (스텝: {step_count})")
    
    env.close()
    
    # 통계 출력
    print("\n" + "="*70)
    print("보상 구성 요소 분석 결과")
    print("="*70)
    
    # 전체 평균 계산
    all_forward = []
    all_ctrl = []
    all_healthy = []
    all_total = []
    all_velocity = []
    
    for ep_data in all_episodes_data:
        all_forward.extend(ep_data['forward_reward'])
        all_ctrl.extend(ep_data['ctrl_cost'])
        all_healthy.extend(ep_data['healthy_reward'])
        all_total.extend(ep_data['total_reward'])
        all_velocity.extend(ep_data['x_velocity'])
    
    print(f"\n전체 {num_episodes}개 에피소드 평균:")
    print(f"  Forward Reward (전진 보상):  {np.mean(all_forward):8.3f} (±{np.std(all_forward):.3f})")
    print(f"  Control Cost (제어 비용):    {np.mean(all_ctrl):8.3f} (±{np.std(all_ctrl):.3f})")
    print(f"  Healthy Reward (생존 보상):  {np.mean(all_healthy):8.3f} (±{np.std(all_healthy):.3f})")
    print(f"  Total Reward (총 보상):      {np.mean(all_total):8.3f} (±{np.std(all_total):.3f})")
    print(f"  X Velocity (전진 속도):      {np.mean(all_velocity):8.3f} m/s (±{np.std(all_velocity):.3f})")
    
    print("="*70)
    print("보상 분석 완료!\n")

# ---------------------------------------------
# 5. 실행
# ---------------------------------------------
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Walker2d 장시간 속도 분석')
    parser.add_argument('--mode', type=str, default='speed',
                       choices=['speed', 'reward'],
                       help='실행 모드: speed (속도 분석), reward (보상 분석)')
    parser.add_argument('--model', type=str, 
                       default='./walker2d_speed_boost_logs/best_model/best_model.zip',
                       help='모델 파일 경로')
    parser.add_argument('--norm', type=str,
                       default='./walker2d_speed_boost_logs/vec_normalize_best.pkl',
                       help='VecNormalize 통계 파일 경로')
    parser.add_argument('--steps', type=int, default=50000,
                       help='총 실행 스텝 수')
    parser.add_argument('--warmup', type=int, default=500,
                       help='통계 수집 시작 전 워밍업 스텝')
    parser.add_argument('--episodes', type=int, default=10,
                       help='보상 분석 시 실행할 에피소드 수')
    parser.add_argument('--plot-folder', type=str, default='./speed_analysis/',
                       help='분석 플롯 저장 폴더')
    parser.add_argument('--render', action='store_true',
                       help='렌더링 활성화')
    
    args = parser.parse_args()
    
    if args.mode == 'speed':
        # 속도 분석 실행
        analyze_long_run_speed(
            model_path=args.model,
            vec_normalize_path=args.norm,
            total_steps=args.steps,
            warmup_steps=args.warmup,
            save_plots=True,
            plot_folder=args.plot_folder,
            render=args.render
        )
    else:  # reward
        # 보상 분석 실행
        analyze_rewards(
            model_path=args.model,
            vec_normalize_path=args.norm,
            num_episodes=args.episodes,
            save_plots=True,
            plot_folder=args.plot_folder
        )