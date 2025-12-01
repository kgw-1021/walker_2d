import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
import numpy as np
import torch
import os
from typing import Optional

# ---------------------------------------------
# 1. 환경 설정 및 장치 확인
# ---------------------------------------------
os.environ['MUJOCO_GL'] = 'glfw'
os.environ['SDL_VIDEODRIVER'] = 'windows'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

# ---------------------------------------------
# 2. CommandAugmentedWalker (학습 때 사용한 환경 래퍼)
# ---------------------------------------------
# 이 클래스는 학습된 모델이 기대하는 18차원 관측값을 생성하기 위해 필수입니다.
class CommandAugmentedWalker(gym.Wrapper):
    """
    관측(Observation)에 '목표 속도(Command)'를 추가하는 환경 래퍼.
    """
    def __init__(self, env, command_low=3.0, final_command_high=3.0, init_command_high=3.0):
        super().__init__(env)
        self.command_low = command_low
        self.final_command_high = final_command_high
        self.curr_max_speed = init_command_high # 테스트 시에는 최대 속도 범위(3.0)로 설정
        self.current_command = 0.0
        
        # 관측 공간 확장: 기존 Obs + Command(1)
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        new_low = np.concatenate([low, np.array([command_low], dtype=np.float32)])
        new_high = np.concatenate([high, np.array([final_command_high], dtype=np.float32)])
        self.observation_space = spaces.Box(low=new_low, high=new_high, dtype=np.float32)

    def set_max_speed(self, speed):
        """커리큘럼 학습 중 사용되는 함수 (테스트 시에는 주로 사용 안함)"""
        self.curr_max_speed = min(speed, self.final_command_high)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # 테스트 시에는 최대 난이도(3.0) 내에서 랜덤 목표 속도 설정
        self.current_command = np.random.uniform(self.command_low, self.curr_max_speed)
        
        # 관측값에 Command 추가 (모델이 기대하는 18차원 생성)
        obs_aug = np.concatenate([obs, np.array([self.current_command], dtype=np.float32)])
        
        # 정보용 dict에 목표 속도 기록
        info["target_velocity"] = self.current_command
        return obs_aug, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # *주의*: 테스트 시각화 코드에서는 보상 함수가 중요하지 않으므로, 
        # 학습 시 사용했던 복잡한 커스텀 보상 대신, 
        # 관측값 확장과 정보 추출만 수행하여 모델이 기대하는 형식을 유지합니다.
        
        x_vel = info.get('x_velocity', 0)
        target_vel = self.current_command
        
        # (테스트를 위해 단순화된 보상 로직을 적용하거나, 복잡한 로직을 가져올 수 있습니다.)
        # 여기서는 모델이 학습된 방식대로 CommandAugmentedWalker의 로직을 따라갑니다.
        velocity_error = np.abs(target_vel - x_vel)
        tracking_reward = -2.0 * velocity_error
        achievement_bonus = 1.5 if velocity_error < 0.3 else 0.0
        survival_reward = 0.05
        ctrl_cost = -0.001 * np.sum(np.square(action))
        
        # 최종 보상은 기록용
        custom_reward = tracking_reward + achievement_bonus + survival_reward + ctrl_cost

        # 관측값에 Command 추가 (모델이 기대하는 18차원 생성)
        obs_aug = np.concatenate([obs, np.array([self.current_command], dtype=np.float32)])
        
        info["target_velocity"] = target_vel
        info["x_velocity"] = x_vel # 속도 정보 추가
        
        return obs_aug, custom_reward, terminated, truncated, info

# ---------------------------------------------
# 3. Best Model 시각화 전용 코드 (수정됨)
# ---------------------------------------------
def visualize_best_model(
    model_path: str = "./walker2d_curriculum_logs/best_model/best_model.zip",
    vec_normalize_path: str = "./walker2d_curriculum_logs/vec_normalize.pkl",
    num_episodes: int = 5,
    render_mode: str = "human",
    deterministic: bool = True,
    save_video: bool = False,
    video_folder: str = "./walker2d_videos/",
):
    """
    저장된 best model을 불러와서 시각화합니다. (커맨드 및 정규화 반영)
    """
    
    print("\n" + "="*70)
    print("Best Model 시각화 시작 (커맨드 추적)")
    print("="*70)
    
    # 모델 및 정규화 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    if not os.path.exists(vec_normalize_path):
        print(f"오류: VecNormalize 통계 파일을 찾을 수 없습니다: {vec_normalize_path}")
        print("정규화 없이 시도합니다. (학습된 모델이 정규화에 의존하면 실패할 수 있음)")
        use_normalization = False
    else:
        use_normalization = True
    
    # 비디오 저장 설정
    if save_video:
        os.makedirs(video_folder, exist_ok=True)
        render_mode = "rgb_array"
    
    # 환경 생성 (학습 환경과 동일하게 CommandAugmentedWalker로 래핑)
    # 테스트 시에는 최종 목표인 3.0 m/s를 최대 속도 범위로 설정합니다.
    vis_env = gym.make("Walker2d-v5", render_mode=render_mode)
    vis_env = CommandAugmentedWalker(vis_env, init_command_high=3.0) 
    
    # 비디오 레코더 설정
    if save_video:
        from gymnasium.wrappers import RecordVideo
        vis_env = RecordVideo(
            vis_env, 
            video_folder=video_folder,
            episode_trigger=lambda x: True, 
            name_prefix="walker2d_command_best"
        )
    
    # 모델 불러오기
    loaded_model = SAC.load(model_path, device=device)
    
    # 정규화 통계 로드 및 적용
    obs_mean = None
    obs_std = None
    epsilon = 1e-8
    
    if use_normalization:
        # VecNormalize 객체를 로드하여 통계값만 추출
        print(f"✓ 정규화 통계 로드: {vec_normalize_path}")
        
        # VecNormalize 로딩을 위해서는 DummyVecEnv가 필요합니다.
        # 로드된 객체에서 통계만 가져와 수동 정규화에 사용합니다.
        temp_env = DummyVecEnv([lambda: gym.make("Walker2d-v5")]) 
        vec_norm = VecNormalize.load(vec_normalize_path, temp_env)
        
        obs_mean = vec_norm.obs_rms.mean
        obs_std = np.sqrt(vec_norm.obs_rms.var + epsilon)
        
        # 임시 환경 닫기
        temp_env.close()

    print(f"✓ 모델 불러오기: {model_path}")
    print(f"✓ 정책 모드: {'결정적 (Deterministic)' if deterministic else '확률적 (Stochastic)'}")
    print(f"✓ 렌더링 모드: {render_mode}")
    print()
    
    # 에피소드 실행
    episode_rewards = []
    episode_velocities = []
    episode_steps = []
    
    for episode in range(num_episodes):
        # 1. 환경 리셋 및 초기 관측값(18차원) 획득
        obs_array, info = vis_env.reset()
        episode_reward = 0
        current_target_vel = info["target_velocity"]
        total_velocity = 0 # 평균 속도를 계산하기 위한 누적 변수
        step_count = 0
        done = False
        
        print(f"에피소드 {episode + 1}/{num_episodes} (목표 속도: {current_target_vel:.2f} m/s) 실행 중...", end="\r")
        
        while not done:
            # 2. 관측값 정규화 (학습 시와 동일하게)
            if use_normalization:
                obs_norm = (obs_array - obs_mean) / obs_std
                # Clip obs, 학습 시 설정했던 값(10.0)과 동일하게 적용
                obs_norm = np.clip(obs_norm, -10.0, 10.0) 
            else:
                obs_norm = obs_array
            
            # 3. 모델 예측 (18차원 입력)
            # SB3의 predict는 numpy 배열 하나를 받으면 내부적으로 배치 차원을 추가합니다.
            action, _states = loaded_model.predict(obs_norm, deterministic=deterministic)
            
            # 4. 환경 스텝
            obs_array, reward, terminated, truncated, info = vis_env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            total_velocity += info['x_velocity']
            step_count += 1
        
        # 에피소드 종료 후 통계 기록
        episode_rewards.append(episode_reward)
        episode_velocities.append(total_velocity / step_count)
        episode_steps.append(step_count)
        
        print(f"에피소드 {episode + 1}/{num_episodes} 완료! [보상: {episode_reward:.2f}, "
              f"목표 속도: {current_target_vel:.2f} m/s, 평균 속도: {total_velocity / step_count:.2f} m/s, 스텝: {step_count}]")
    
    # 최종 통계 출력
    print("\n" + "="*70)
    print("시각화 결과 통계")
    print("="*70)
    print(f"총 에피소드 수: {num_episodes}")
    print(f"평균 보상: {np.mean(episode_rewards):.2f} (±{np.std(episode_rewards):.2f})")
    print(f"평균 달성 속도: {np.mean(episode_velocities):.2f} m/s (±{np.std(episode_velocities):.2f} m/s)")
    print(f"평균 생존 스텝: {np.mean(episode_steps):.1f} (±{np.std(episode_steps):.1f})")
    print(f"최고 보상: {max(episode_rewards):.2f}")
    
    if save_video:
        print(f"\n비디오가 저장되었습니다: {video_folder}")
    
    print("="*70)
    
    # 환경 정리
    vis_env.close()

# ---------------------------------------------
# 4. 실행 예시 (수정된 visualize_best_model 함수 사용)
# ---------------------------------------------
if __name__ == '__main__':
    
    # 학습 시 log_dir이 './walker2d_curriculum_logs/'로 설정되었음을 가정
    visualize_best_model(
        model_path="./walker2d_curriculum_logs/best_model/best_model.zip",
        vec_normalize_path="./walker2d_curriculum_logs/vec_normalize.pkl",
        num_episodes=3,
        save_video=True,
        video_folder="./walker2d_videos_command/", # 커맨드 추적 비디오 폴더명 변경
        render_mode="human", # human 모드에서 비디오 저장은 백그라운드에서 진행됨
        deterministic=True
    )