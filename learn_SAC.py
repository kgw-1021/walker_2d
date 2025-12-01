import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import numpy as np
import torch
import os

# ---------------------------------------------
# 1. 설정 및 초기화
# ---------------------------------------------
log_dir = "./walker2d_curriculum_logs/"
os.makedirs(log_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

# ---------------------------------------------
# 2. 커리큘럼 지원 환경 래퍼
# ---------------------------------------------
class CommandAugmentedWalker(gym.Wrapper):
    """
    커리큘럼 학습을 지원하는 Low-Level Agent 환경 래퍼
    - 넘어짐 방지를 위해 보상 함수를 수정했습니다.
    """
    def __init__(self, env, command_low=0.0, final_command_high=3.0, init_command_high=0.5):
        super().__init__(env)
        self.command_low = command_low
        self.final_command_high = final_command_high # 최종 목표 (3.0 m/s)
        
        # [커리큘럼] 현재 단계의 최대 속도 제한 (초기값: 0.5 m/s)
        self.curr_max_speed = init_command_high 
        
        self.current_command = 0.0
        
        # 관측 공간 확장
        low = self.env.observation_space.low
        high = self.env.observation_space.high
        new_low = np.concatenate([low, np.array([command_low], dtype=np.float32)])
        new_high = np.concatenate([high, np.array([final_command_high], dtype=np.float32)])
        self.observation_space = spaces.Box(low=new_low, high=new_high, dtype=np.float32)

    def set_max_speed(self, speed):
        """외부에서 난이도(최대 속도)를 조절하기 위한 함수"""
        self.curr_max_speed = min(speed, self.final_command_high)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # [커리큘럼] 현재 허용된 최대 속도(curr_max_speed) 내에서 목표 랜덤 설정
        self.current_command = np.random.uniform(self.command_low, self.curr_max_speed)
        
        obs_aug = np.concatenate([obs, np.array([self.current_command], dtype=np.float32)])
        info["target_velocity"] = self.current_command
        return obs_aug, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Walker2d의 Obs[0]은 몸통 Z좌표(높이)입니다.
        torso_height = obs[0]
        x_vel = info.get('x_velocity', obs[8])
        target_vel = self.current_command
        
        # ------------------- 보상 설계 수정 (넘어짐 방지 강화) -------------------
        velocity_error = np.abs(target_vel - x_vel)
        
        # 1. 추적 보상
        tracking_reward = -2.0 * velocity_error
        
        # 2. 달성 보너스 
        achievement_bonus = 1.5 if velocity_error < 0.3 else 0.0
        
        # 3. 생존 보상 (기존 0.05 -> 1.0 으로 강화)
        survival_reward = 1.0 
        
        # 4. 제어 비용 (기존 -0.001 -> -0.1 로 강화, 불안정한 움직임 방지)
        ctrl_cost = -0.1 * np.sum(np.square(action)) 
        
        # 5. **몸통 높이 페널티 (새로 추가)**: 높이가 0.8m 미만일 때 강력한 페널티
        height_penalty = 0.0
        MIN_HEIGHT_THRESHOLD = 0.8
        PENALTY_MULTIPLIER = 10.0
        
        if torso_height < MIN_HEIGHT_THRESHOLD:
            # 높이가 낮아질수록 페널티가 급격히 커짐
            height_penalty = -PENALTY_MULTIPLIER * (MIN_HEIGHT_THRESHOLD - torso_height) 
            
        custom_reward = (
            tracking_reward + 
            achievement_bonus + 
            survival_reward + 
            ctrl_cost +
            height_penalty # 새로운 높이 페널티 추가
        )

        obs_aug = np.concatenate([obs, np.array([self.current_command], dtype=np.float32)])
        
        info["target_velocity"] = target_vel
        info["velocity_error"] = velocity_error
        info["curr_max_speed"] = self.curr_max_speed # 현재 난이도 정보 기록
        
        return obs_aug, custom_reward, terminated, truncated, info

# ---------------------------------------------
# 3. 커리큘럼 콜백 (핵심 로직)
# ---------------------------------------------
class CurriculumCallback(BaseCallback):
    """
    학습 성과를 모니터링하다가 잘하면 난이도(최대 속도)를 올리는 콜백
    """
    def __init__(self, check_freq=5000, error_threshold=0.4, step_size=0.5, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.error_threshold = error_threshold # 이 오차보다 줄어들면 난이도 상승
        self.step_size = step_size # 한 번에 올릴 속도 (m/s)
        self.error_buffer = []
        self.current_level = 0.5 # 초기 난이도

    def _on_step(self) -> bool:
        # 현재 스텝의 velocity_error 정보를 수집
        infos = self.locals.get("infos", [])
        for info in infos:
            if "velocity_error" in info:
                self.error_buffer.append(info["velocity_error"])
        
        # 일정 주기마다 평가 및 난이도 조절
        if self.n_calls % self.check_freq == 0:
            if len(self.error_buffer) > 0:
                mean_error = np.mean(self.error_buffer)
                
                if self.verbose > 0:
                    print(f"\n[Curriculum] Step {self.n_calls}: 평균 속도 오차 = {mean_error:.3f} (목표: < {self.error_threshold})")
                    print(f"[Curriculum] 현재 최대 속도 레벨: {self.current_level:.1f} m/s")

                # 목표 달성 시 난이도 상승
                if mean_error < self.error_threshold and self.current_level < 3.0:
                    self.current_level = min(self.current_level + self.step_size, 3.0)
                    
                    # 훈련 환경(env)에 새로운 난이도 적용
                    # DummyVecEnv 내부의 원본 환경들에 접근하여 값 설정
                    env = self.training_env
                    # Unwrap하여 CommandAugmentedWalker 찾기
                    # (VecNormalize -> DummyVecEnv -> Monitor -> CommandAugmentedWalker 순서)
                    # 가장 확실한 방법: get_attr이나 env_method 사용
                    env.env_method("set_max_speed", self.current_level)
                    
                    print(f"🎉 성과 달성! 난이도 상승 -> 최대 속도: {self.current_level:.1f} m/s 로 변경됨.\n")
                
                # 버퍼 초기화
                self.error_buffer = []
        
        return True

# ---------------------------------------------
# 4. 환경 생성 및 실행
# ---------------------------------------------
def make_env():
    env = gym.make("Walker2d-v5") 
    # 초기 난이도 0.5부터 시작, 최종 3.0까지
    env = CommandAugmentedWalker(env, command_low=0.0, final_command_high=3.0, init_command_high=0.5)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    # 환경 생성
    env = DummyVecEnv([make_env for _ in range(1)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    print(f"\n커리큘럼 SAC 학습 시작 (Device: {device})")
    print("초기 목표 속도 범위: 0.0 ~ 0.5 m/s")
    
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        ent_coef='auto',
        verbose=1,
        tensorboard_log=log_dir + "tensorboard/",
        device=device,
    )

    # 콜백 설정
    # 1. 커리큘럼 콜백: 5000 스텝마다 검사, 오차가 0.35 미만이면 난이도 0.5씩 증가
    curriculum_cb = CurriculumCallback(check_freq=5000, error_threshold=0.35, step_size=0.5)
    
    # 2. 평가 콜백 (기존 유지)
    eval_env = DummyVecEnv([make_env for _ in range(1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10., training=False)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "best_model/",
        log_path=log_dir + "eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True
    )

    # 학습 시작
    model.learn(total_timesteps=1000000, callback=[curriculum_cb, eval_callback], progress_bar=True)

    model.save(log_dir + "low_level_curriculum_final")
    env.save(log_dir + "vec_normalize.pkl")
    print("학습 완료.")

    # ---------------------------------------------
    # 테스트 (최종 성능 확인)
    # ---------------------------------------------
    print("\n최종 테스트 시작 (Max Speed 3.0)...")
    test_env_base = gym.make("Walker2d-v5", render_mode="human")
    # 테스트 시에는 최대 난이도로 설정
    test_env = CommandAugmentedWalker(test_env_base, command_low=0.0, final_command_high=3.0, init_command_high=3.0)
    
    obs, info = test_env.reset()
    
    # 정규화 통계 로드
    obs_mean = env.obs_rms.mean
    obs_var = env.obs_rms.var
    epsilon = 1e-8

    for i in range(1000):
        # 수동 정규화
        obs_norm = (obs - obs_mean) / np.sqrt(obs_var + epsilon)
        obs_norm = np.clip(obs_norm, -10, 10)
        
        action, _ = model.predict(obs_norm, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        print(f"Step {i}: 목표={info['target_velocity']:.2f}, 현재={info['x_velocity']:.2f}")
        
        if terminated or truncated:
            obs, info = test_env.reset()

    test_env.close()