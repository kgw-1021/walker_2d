import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch
import os

# ---------------------------------------------
# 1. 설정 및 초기화
# ---------------------------------------------
log_dir = "./walker2d_speed_boost_logs/"
os.makedirs(log_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

# ---------------------------------------------
# 2. 속도 증가 보상 래퍼
# ---------------------------------------------
class SpeedBoostWrapper(gym.Wrapper):
    """
    기본 보상은 그대로 유지하고, 속도가 증가할수록 작은 보너스를 추가
    
    보상 구조:
    - 기본 보상: forward_reward - ctrl_cost + healthy_reward (그대로 유지)
    - 추가 보상: 속도에 비례하는 작은 보너스
    """
    def __init__(self, env, speed_bonus_coef=0.1):
        super().__init__(env)
        self.speed_bonus_coef = speed_bonus_coef
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 현재 속도 가져오기
        x_velocity = info.get('x_velocity', obs[8])
        
        # 속도 증가 보너스 (속도에 비례)
        # 기준점: 3.8 m/s (vanilla SAC의 평균 속도)
        baseline_speed = 3.8
        
        # 3.8 이상일 때만 보너스 (초반 학습 안정성을 위해)
        if x_velocity > baseline_speed:
            speed_bonus = self.speed_bonus_coef * (x_velocity - baseline_speed)
        else:
            speed_bonus = 0.0
        
        # 기본 보상 + 속도 보너스
        modified_reward = reward + speed_bonus
        
        # 디버깅용 정보 추가
        info['speed_bonus'] = speed_bonus
        info['original_reward'] = reward
        info['modified_reward'] = modified_reward
        
        return obs, modified_reward, terminated, truncated, info

# ---------------------------------------------
# 3. VecNormalize 저장 콜백
# ---------------------------------------------
class SaveVecNormalizeCallback(BaseCallback):
    """최고 모델 저장 시 VecNormalize 통계를 함께 저장"""
    def __init__(self, vec_normalize_env, save_path):
        super().__init__()
        self.vec_normalize_env = vec_normalize_env
        self.save_path = save_path
    
    def _on_step(self):
        self.vec_normalize_env.save(self.save_path)
        print(f"VecNormalize 통계 저장 완료: {self.save_path}")
        return True

# ---------------------------------------------
# 4. 속도 모니터링 콜백
# ---------------------------------------------
class SpeedMonitorCallback(BaseCallback):
    """
    학습 중 속도와 보상을 모니터링
    """
    def __init__(self, check_freq=10000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.velocities = []
        self.speed_bonuses = []
        self.original_rewards = []
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "x_velocity" in info:
                self.velocities.append(info["x_velocity"])
            if "speed_bonus" in info:
                self.speed_bonuses.append(info["speed_bonus"])
            if "original_reward" in info:
                self.original_rewards.append(info["original_reward"])
        
        if self.n_calls % self.check_freq == 0:
            if len(self.velocities) > 0:
                mean_velocity = np.mean(self.velocities)
                mean_speed_bonus = np.mean(self.speed_bonuses) if len(self.speed_bonuses) > 0 else 0
                mean_original = np.mean(self.original_rewards) if len(self.original_rewards) > 0 else 0
                
                if self.verbose > 0:
                    print(f"\n[Speed Monitor] Step {self.n_calls}:")
                    print(f"  평균 속도: {mean_velocity:.3f} m/s")
                    print(f"  평균 속도 보너스: {mean_speed_bonus:.3f}")
                    print(f"  평균 원본 보상: {mean_original:.3f}")
                
                # 버퍼 초기화
                self.velocities = []
                self.speed_bonuses = []
                self.original_rewards = []
        
        return True

# ---------------------------------------------
# 5. 환경 생성 함수
# ---------------------------------------------
def make_env(speed_bonus_coef=0.1):
    """속도 증가 보상이 추가된 Walker2d-v5 환경 생성"""
    env = gym.make("Walker2d-v5")
    env = SpeedBoostWrapper(env, speed_bonus_coef=speed_bonus_coef)
    env = Monitor(env)
    return env

# ---------------------------------------------
# 6. 학습 실행
# ---------------------------------------------
if __name__ == "__main__":
    # 속도 보너스 계수 설정
    # 작은 값부터 시작 (0.05 ~ 0.2 정도 권장)
    SPEED_BONUS_COEF = 0.1
    
    print(f"\n속도 증가 보상 SAC 학습 시작 (Device: {device})")
    print(f"속도 보너스 계수: {SPEED_BONUS_COEF}")
    print(f"보상 설계: 기본 보상 + {SPEED_BONUS_COEF} * (속도 - 3.8)")
    print("="*70 + "\n")
    
    # 학습 환경 생성 (병렬 환경)
    train_env = SubprocVecEnv([lambda: make_env(SPEED_BONUS_COEF) for _ in range(8)])
    train_env = VecNormalize(
        train_env, 
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )

    # SAC 모델 생성
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        target_entropy='auto',
        verbose=1,
        tensorboard_log=log_dir + "tensorboard/",
        device=device,
    )

    # 평가 환경 생성
    eval_env = DummyVecEnv([lambda: make_env(SPEED_BONUS_COEF)])
    eval_env = VecNormalize(
        eval_env, 
        norm_obs=True, 
        norm_reward=False,
        clip_obs=10.0,
        training=False
    )

    # VecNormalize 저장 콜백
    save_norm_cb = SaveVecNormalizeCallback(
        train_env, 
        log_dir + "vec_normalize_best.pkl"
    )

    # 평가 콜백
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "best_model/",
        log_path=log_dir + "eval/",
        eval_freq=2000,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=save_norm_cb
    )

    # 속도 모니터링 콜백
    speed_monitor = SpeedMonitorCallback(check_freq=10000)

    # 학습 시작
    print("학습을 시작합니다...")
    model.learn(
        total_timesteps=5000000,
        callback=[speed_monitor, eval_callback],
        progress_bar=True
    )

    # 최종 모델 저장
    model.save(log_dir + "walker2d_speed_boost_final")
    train_env.save(log_dir + "vec_normalize_final.pkl")
    print("\n학습 완료!")

    # ---------------------------------------------
    # 7. 테스트 (학습된 모델 평가)
    # ---------------------------------------------
    print("\n최종 성능 테스트 시작...")
    
    # 테스트 환경 생성
    def make_test_env():
        env = gym.make("Walker2d-v5", render_mode="human")
        env = SpeedBoostWrapper(env, speed_bonus_coef=SPEED_BONUS_COEF)
        return Monitor(env)
    
    test_vec_env = DummyVecEnv([make_test_env])
    
    # 학습 시 저장한 정규화 통계 로드
    test_vec_env = VecNormalize.load(
        log_dir + "vec_normalize_best.pkl",
        test_vec_env
    )
    
    # 테스트 모드 설정
    test_vec_env.training = False
    test_vec_env.norm_reward = False

    # 테스트 실행
    obs = test_vec_env.reset()
    episode_rewards = []
    episode_velocities = []
    episode_speed_bonuses = []
    episode_reward = 0
    episode_velocity_sum = 0
    episode_speed_bonus_sum = 0
    episode_count = 0
    step_count = 0

    for step in range(3000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_vec_env.step(action)
        
        episode_reward += reward[0]
        if 'x_velocity' in info[0]:
            episode_velocity_sum += info[0]['x_velocity']
        if 'speed_bonus' in info[0]:
            episode_speed_bonus_sum += info[0]['speed_bonus']
        step_count += 1
        
        if done[0]:
            episode_count += 1
            avg_velocity = episode_velocity_sum / step_count if step_count > 0 else 0
            avg_speed_bonus = episode_speed_bonus_sum / step_count if step_count > 0 else 0
            
            episode_rewards.append(episode_reward)
            episode_velocities.append(avg_velocity)
            episode_speed_bonuses.append(avg_speed_bonus)
            
            print(f"\nEpisode {episode_count} 종료:")
            print(f"  총 보상: {episode_reward:.2f}")
            print(f"  평균 속도: {avg_velocity:.2f} m/s")
            print(f"  평균 속도 보너스: {avg_speed_bonus:.3f}")
            print(f"  에피소드 길이: {step_count}")
            
            episode_reward = 0
            episode_velocity_sum = 0
            episode_speed_bonus_sum = 0
            step_count = 0
            obs = test_vec_env.reset()
    
    if len(episode_rewards) > 0:
        print(f"\n=== 테스트 요약 ===")
        print(f"총 에피소드 수: {len(episode_rewards)}")
        print(f"평균 보상: {np.mean(episode_rewards):.2f}")
        print(f"평균 속도: {np.mean(episode_velocities):.2f} m/s")
        print(f"평균 속도 보너스: {np.mean(episode_speed_bonuses):.3f}")
        print(f"최고 속도: {max(episode_velocities):.2f} m/s")
    
    test_vec_env.close()
    print("\n테스트 완료!")