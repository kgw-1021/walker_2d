import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization, SubprocVecEnv
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
        self.final_command_high = final_command_high 
        
    
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
        tracking_reward = -2.0 * (velocity_error ** 2)

        # 2. 달성 보너스
        achievement_bonus = 5.0 if velocity_error < 0.2 else 0.0

        # 4. 생존 보상 (최소화)
        survival_reward = 0.5

        # 5. 제어 비용 (약하게)
        ctrl_cost = -0.001 * np.sum(np.square(action))

        # 6. 높이 페널티 (온화하게)
        height_penalty = 0.0
        if torso_height < 1.0:
            height_penalty = -2.0 * max(0, (1.0 - torso_height) ** 2)

        custom_reward = (
            tracking_reward + 
            achievement_bonus + 
            survival_reward + 
            ctrl_cost +
            height_penalty
        )

        obs_aug = np.concatenate([obs, np.array([self.current_command], dtype=np.float32)])
        
        info["target_velocity"] = target_vel
        info["velocity_error"] = velocity_error
        info["curr_max_speed"] = self.curr_max_speed # 현재 난이도 정보 기록
        
        return obs_aug, custom_reward, terminated, truncated, info

class SaveVecNormalizeCallback(BaseCallback):
    """최고 모델 저장 시 VecNormalize 통계를 함께 저장"""
    def __init__(self, vec_normalize_env, save_path):
        super().__init__()
        self.vec_normalize_env = vec_normalize_env
        self.save_path = save_path
    
    # 이 메서드는 EvalCallback이 성능 개선을 감지할 때마다 호출됩니다.
    def _on_step(self):
        self.vec_normalize_env.save(self.save_path)
        print(f"VecNormalize 통계 저장 완료: {self.save_path}")
        return True

# ---------------------------------------------
# 3. 커리큘럼 콜백 (핵심 로직)
# ---------------------------------------------
class CurriculumCallback(BaseCallback):
    """
    학습 성과를 모니터링하다가 잘하면 난이도(최대 속도)를 올리는 콜백
    난이도 상승 후, 에피소드 길이가 급격히 짧아지면 난이도를 낮추는 롤백 기능을 추가
    """
    def __init__(self, check_freq=5000, error_threshold=0.35, step_size=0.5, 
                min_ep_len=800, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.error_threshold = error_threshold
        self.step_size = step_size 
        self.min_ep_len = min_ep_len # 난이도 상승을 위한 최소 평균 에피소드 길이
        self.error_buffer = []
        self.ep_len_buffer = []
        self.current_level = 0.5
        self.previous_level = 0.5

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "velocity_error" in info:
                self.error_buffer.append(info["velocity_error"])
            # 'episode' 키는 에피소드가 종료될 때만 생성됩니다.
            if "episode" in info:
                self.ep_len_buffer.append(info["episode"]["l"]) # l: episode length (길이)
        
        if self.n_calls % self.check_freq == 0:
            if len(self.error_buffer) > 0 and len(self.ep_len_buffer) > 0:
                mean_error = np.mean(self.error_buffer)
                mean_ep_len = np.mean(self.ep_len_buffer)
                
                if self.verbose > 0:
                    print(f"\n[Curriculum] Step {self.n_calls}: ")
                    print(f"-> 평균 속도 오차 = {mean_error:.3f} (목표: < {self.error_threshold})")
                    print(f"-> 평균 에피소드 길이 = {mean_ep_len:.1f} (목표: > {self.min_ep_len})")
                    print(f"-> 현재 최대 속도 레벨: {self.current_level:.1f} m/s")

                # 상승과 롤백을 명확히 분리
                level_changed = False
                
                # 1. 먼저 롤백 체크 (성능 저하 시)
                if (mean_error > self.error_threshold * 1.5) or (mean_ep_len < self.min_ep_len * 0.6):
                    if self.current_level > 0.5:
                        self.current_level = max(self.current_level - self.step_size, 0.5)
                        self.training_env.env_method("set_max_speed", self.current_level)
                        print(f"성능 저하 감지! 난이도 하향 -> 최대 속도: {self.current_level:.1f} m/s\n")
                        level_changed = True
                
                # 2. 롤백이 없었고, 성능이 좋으면 상승
                elif (mean_error < self.error_threshold) and (mean_ep_len >= self.min_ep_len) and (self.current_level < 3.0):
                    self.previous_level = self.current_level
                    self.current_level = min(self.current_level + self.step_size, 3.0)
                    self.training_env.env_method("set_max_speed", self.current_level)
                    print(f"성과 달성! 난이도 상승 -> 최대 속도: {self.current_level:.1f} m/s\n")
                    level_changed = True
                
                # 3. 난이도가 변경되었을 때만 버퍼 리셋
                if level_changed:
                    self.error_buffer = []
                    self.ep_len_buffer = []

        return True

class SyncNormalizationCallback(BaseCallback):
    """평가 전에 정규화 통계를 동기화"""
    def __init__(self, training_env, eval_env):
        super().__init__()
        self._training_env_to_sync = training_env
        self._eval_env_to_sync = eval_env
    
    def _on_step(self):
        # 평가 직전 통계 동기화
        sync_envs_normalization(self._training_env_to_sync, self._eval_env_to_sync)
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
    env = SubprocVecEnv([make_env for _ in range(8)])
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
    
    save_norm_cb = SaveVecNormalizeCallback(env, log_dir + "vec_normalize_best.pkl")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "best_model/",
        log_path=log_dir + "eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        callback_on_new_best=save_norm_cb
    )

    sync_callback = SyncNormalizationCallback(env, eval_env)

    # 학습 시작
    model.learn(total_timesteps=1000000, callback=[curriculum_cb, sync_callback, eval_callback], progress_bar=True)

    model.save(log_dir + "low_level_curriculum_final")
    env.save(log_dir + "vec_normalize.pkl")
    print("학습 완료.")

    # ---------------------------------------------
    # 테스트 (최종 성능 확인)
    # ---------------------------------------------
    print("\n최종 테스트 시작 (Max Speed 3.0)...")
    # 1. 테스트 환경을 VecEnv로 감싸기
    def make_test_env():
        env_base = gym.make("Walker2d-v5", render_mode="human")
        env_wrapped = CommandAugmentedWalker(
            env_base, 
            command_low=0.0, 
            final_command_high=3.0, 
            init_command_high=3.0
        )
        return Monitor(env_wrapped)

    test_vec_env = DummyVecEnv([make_test_env])

    # 2. 학습 때 저장한 정규화 통계 불러오기
    test_vec_env = VecNormalize.load(
        log_dir + "vec_normalize.pkl",
        test_vec_env
    )

    # 3. 테스트 모드 설정 (중요!)
    test_vec_env.training = False  # 통계 업데이트 중지
    test_vec_env.norm_reward = False  # 보상 정규화 비활성화

    # 4. 테스트 실행
    obs = test_vec_env.reset()

    for i in range(1000):
        # VecNormalize가 자동으로 정규화해줌
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_vec_env.step(action)
        
        # VecEnv는 배치 형태로 반환하므로 인덱싱 필요
        if done[0]:
            print(f"Step {i}: 에피소드 종료")
            print(f"  - 목표 속도: {info[0]['target_velocity']:.2f} m/s")
            print(f"  - 달성 속도: {info[0]['x_velocity']:.2f} m/s")
            print(f"  - 오차: {info[0]['velocity_error']:.3f}")
            obs = test_vec_env.reset()

    test_vec_env.close()