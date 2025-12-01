# hrl_sac_distill_refactor.py
import os
import copy
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from typing import Optional
from collections import deque
import gymnasium.spaces as spaces

# -----------------------------
# Config
# -----------------------------
ENV_ID = "Walker2d-v5"
NUM_ENVS = 8
SEED = 42
TOTAL_TIMESTEPS_LOW = 1_000_000   # 예: 실험 목적에 맞게 조정
TOTAL_TIMESTEPS_HIGH = 500_000
LOW_LEVEL_DURATION = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
DISTILLATION_WEIGHT = 1.0
TEACHER_PATH = None
LOG_DIR = "./hrl_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 학습 중단 기준
TARGET_MIN_VELOCITY = 3.0
EVALUATION_WINDOW = 1000
CONSECUTIVE_SUCCESS = 5

# -----------------------------
# Utility: env factory that's SB3/SubprocVecEnv friendly
# -----------------------------
def make_base_env(env_id, seed, rank=0):
    def _init():
        env = gym.make(env_id)
        # Gymnasium API: reset with seed ensures deterministic spawn
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# -----------------------------
# Reward wrapper for maximum speed (cleaned)
# -----------------------------
class MaxSpeedRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.velocity_history = deque(maxlen=EVALUATION_WINDOW)

    def reset(self, **kwargs):
        self.velocity_history.clear()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Walker2d forward velocity
        # obs[8]은 일반적으로 Mujoco 환경에서 x-축(전진) 속도를 나타냅니다.
        x_vel = obs[8] if len(obs) > 8 else 0.0

        # Track for min velocity calculation
        self.velocity_history.append(x_vel)

        # ----------------------
        # New Scaled Reward (Focus on Forward Momentum)
        # ----------------------
        
        # 1. 전진 속도 보상 (Primary Reward)
        # 속도가 높을수록 선형적으로 높은 보상을 부여합니다. 가중치 1.0으로 중요도 극대화.
        forward_reward = x_vel * 2.0 
        
        # 2. 생존 보상 (Survival Reward - 최소화)
        # 가만히 서 있는 것만으로는 보상을 얻지 못하도록 0.01로 극단적으로 낮춥니다.
        survival_reward = 0.001 
        
        # 3. 최소 속도 보너스 (Min Velocity Bonus - 중요한 보너스)
        # 일정 스텝 이상 진행 후, 최소 속도가 음수가 아닌 경우에만 보너스를 제공하여 
        # 뒤로 걷는 것을 방지하고 꾸준한 전진을 유도합니다.
        min_vel_bonus = 0.0
        if len(self.velocity_history) >= 50:
            # 최소 속도를 0.0 미만으로 제한하지 않아 음수 속도일 때 페널티 효과도 반영
            current_min_velocity = min(self.velocity_history)
            
            # 최소 속도가 양수인 경우, 그 속도에 비례한 보너스를 줍니다. (가중치 0.5)
            # 목표 속도(3.0)에 가까운 최소 속도를 유지하는 것에 크게 기여합니다.
            min_vel_bonus = max(current_min_velocity, 0.0) * 0.5 

        # 4. 행동 페널티 (Action Penalty - 유지)
        # 불필요하게 큰 액션을 사용하는 것을 방지합니다.
        action_penalty = -0.001 * np.sum(np.square(action))

        # 5. 종료 페널티 (Termination Penalty - 감소)
        # 넘어져서 종료될 경우 페널티를 주지만, 학습 초기에 소극적이 되지 않도록 강도를 낮춥니다.
        termination_penalty = 0.0
        if terminated or truncated:
            # Walker2d 환경은 일반적으로 넘어지면 terminated=True가 됩니다.
            # 속도가 음수가 아닐 때 넘어진 경우만 페널티를 줄 수도 있지만, 일단 단순하게 적용합니다.
            termination_penalty = -0.01

        new_reward = (forward_reward + 
                      survival_reward + 
                      min_vel_bonus + 
                      action_penalty +
                      termination_penalty)

        info["x_velocity"] = x_vel
        info["min_velocity"] = min(self.velocity_history) if len(self.velocity_history) > 0 else 0.0
        info["mean_velocity"] = np.mean(self.velocity_history) if len(self.velocity_history) > 0 else 0.0

        return obs, new_reward, terminated, truncated, info
# -----------------------------
# LowLevelEnv
# -----------------------------
class LowLevelEnv(gym.Env):
    """
    Low level env: base_env wrapped by MaxSpeedRewardWrapper,
    observation augmented with a command vector (e.g., target speed).
    IMPORTANT: command is meaningful: low-level reward can depend on how well it tracks the command.
    """
    def __init__(self, base_env_ctor, command_dim=1, command_low=0.0, command_high=3.0):
        # base_env_ctor: callable returning a gym env
        self.base_env = MaxSpeedRewardWrapper(base_env_ctor())
        self.command_dim = command_dim
        self.cmd_low = float(command_low)
        self.cmd_high = float(command_high)

        base_obs_space = self.base_env.observation_space
        # concatenate limits and ensure float32
        low_obs = np.concatenate([np.asarray(base_obs_space.low, dtype=np.float32),
                                  np.ones(command_dim, dtype=np.float32) * self.cmd_low])
        high_obs = np.concatenate([np.asarray(base_obs_space.high, dtype=np.float32),
                                   np.ones(command_dim, dtype=np.float32) * self.cmd_high])
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.action_space = self.base_env.action_space
        self._np_random = None
        self.current_command = np.zeros((self.command_dim,), dtype=np.float32)
        self._last_obs = None

    def seed(self, seed=None):
        self._np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        obs, info = self.base_env.reset(seed=seed, options=options)
        # sample a command for this episode (high-level will override during HRL runs)
        self.current_command = np.asarray(self._np_random.uniform(self.cmd_low, self.cmd_high, size=(self.command_dim,))
                                          if self._np_random is not None else
                                          np.random.uniform(self.cmd_low, self.cmd_high, size=(self.command_dim,)),
                                          dtype=np.float32)
        self._last_obs = obs
        obs_aug = np.concatenate([obs.astype(np.float32), self.current_command.astype(np.float32)])
        return obs_aug, info

    def step(self, action):
        # clip action into environment action space
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        obs, reward, terminated, truncated, info = self.base_env.step(action)
        self._last_obs = obs
        obs_aug = np.concatenate([obs.astype(np.float32), self.current_command.astype(np.float32)])
        return obs_aug, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.base_env.render(mode=mode)

    def close(self):
        return self.base_env.close()

# -----------------------------
# HighLevelEnv
# -----------------------------
class HighLevelEnv(gym.Env):
    """
    High-level env: action = command vector (e.g., target speed).
    On step(), it executes low_agent for low_level_duration steps with the given command.
    IMPORTANT: do NOT reset the base env inside step(). Reset only in reset().
    """
    def __init__(self, base_env_ctor, low_agent: SAC, command_dim=1, low_level_duration=LOW_LEVEL_DURATION):
        self.base_env_ctor = base_env_ctor
        self.low_agent = low_agent
        self.command_dim = command_dim
        self.low_level_duration = low_level_duration
        # we create a temporary base just for space introspection, then close
        tmp = MaxSpeedRewardWrapper(base_env_ctor())
        self.observation_space = tmp.observation_space
        tmp.close()
        self.action_space = spaces.Box(low=np.array([-1.0]*command_dim, dtype=np.float32),
                                       high=np.array([1.0]*command_dim, dtype=np.float32),
                                       dtype=np.float32)
        self.base_env = None
        self._np_random = None
        self._last_obs = None

    def seed(self, seed=None):
        self._np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.base_env = MaxSpeedRewardWrapper(self.base_env_ctor())
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs.astype(np.float32), info

    def step(self, action):
            # clip action into environment action space
            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, self.action_space.low, self.action_space.high)

            # 1. Base Environment Step
            obs, reward, terminated, truncated, info = self.base_env.step(action)
            self._last_obs = obs
            
            # 2. Command Tracking Penalty (HRL 핵심)
            target_vel = self.current_command[0] # Low-Level Agent의 목표 속도
            x_vel = info.get("x_velocity", obs[8] if len(obs) > 8 else 0.0)
            
            # 목표 속도와 현재 속도의 차이에 비례하는 페널티를 부과합니다.
            # 가중치 0.8: 추적 실패에 대한 페널티를 높여 명령을 따르도록 강제합니다.
            tracking_penalty = 0.8 * np.abs(target_vel - x_vel) 

            # 3. 최종 보상 계산
            # MaxSpeedRewardWrapper에서 계산된 보상에서 추적 페널티를 차감합니다.
            new_reward = reward - tracking_penalty
            
            # 4. Observation 확장 및 반환
            obs_aug = np.concatenate([obs.astype(np.float32), self.current_command.astype(np.float32)])
            
            # info 딕셔너리에 LowLevelEnv 관련 정보 추가 (선택 사항)
            info["target_velocity"] = target_vel

            return obs_aug, new_reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.base_env:
            return self.base_env.render(mode=mode)

    def close(self):
        if self.base_env:
            self.base_env.close()

# -----------------------------
# Callbacks (Evaluation / Distillation) - robust to shapes
# -----------------------------
class EvaluationCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_min_velocity = -np.inf
        self.success_count = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        min_velocities = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            velocity_buffer = deque(maxlen=EVALUATION_WINDOW)
            done = False
            step_count = 0
            while not done and step_count < EVALUATION_WINDOW:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                if "x_velocity" in info:
                    velocity_buffer.append(info["x_velocity"])
                elif isinstance(obs, np.ndarray) and obs.size > 8:
                    velocity_buffer.append(float(obs[8]))
                step_count += 1
            if len(velocity_buffer) > 0:
                min_velocities.append(float(min(velocity_buffer)))

        avg_min_velocity = float(np.mean(min_velocities)) if len(min_velocities) > 0 else 0.0

        if self.verbose:
            print("\n" + "="*50)
            print(f"Evaluation at step {self.n_calls}")
            print(f"Average minimum velocity over {EVALUATION_WINDOW} steps: {avg_min_velocity:.3f}")
            print(f"Best so far: {self.best_min_velocity:.3f}")
            print(f"Target: {TARGET_MIN_VELOCITY:.3f}")
            print(f"Success count: {self.success_count}/{CONSECUTIVE_SUCCESS}")
            print("="*50 + "\n")

        if avg_min_velocity > self.best_min_velocity:
            self.best_min_velocity = avg_min_velocity
            # save model
            try:
                self.model.save(os.path.join(LOG_DIR, "best_model"))
                if self.verbose:
                    print(f"Saved new best model (min velocity {self.best_min_velocity:.3f})")
            except Exception as e:
                print("Warning: failed to save best model:", e)

        if avg_min_velocity >= TARGET_MIN_VELOCITY:
            self.success_count += 1
            if self.success_count >= CONSECUTIVE_SUCCESS:
                print("TARGET ACHIEVED. Stopping training.")
                return False
        else:
            self.success_count = 0

        return True

class DistillationCallback(BaseCallback):
    def __init__(self, teacher_model: Optional[SAC], distill_weight: float = 1.0, every_n_calls:int=1000, verbose=0):
        super().__init__(verbose)
        self.teacher = teacher_model
        self.weight = distill_weight
        self.every_n_calls = every_n_calls
        self._calls = 0
        if self.teacher is not None:
            for p in self.teacher.policy.parameters():
                p.requires_grad = False

    def _on_step(self) -> bool:
        self._calls += 1
        if self.teacher is None or (self._calls % self.every_n_calls != 0):
            return True

        model: SAC = self.model
        rb = model.replay_buffer
        if rb is None or rb.size() < BATCH_SIZE:
            return True

        samples = rb.sample(BATCH_SIZE, env=model.get_env())
        obs = samples.observations  # numpy array
        device = torch.device(DEVICE)
        obs_t = torch.as_tensor(obs).to(device)

        with torch.no_grad():
            teacher_dist = self.teacher.policy.get_distribution(obs_t)
        student_dist = model.policy.get_distribution(obs_t)

        sampled_actions = teacher_dist.sample()
        teacher_logp = teacher_dist.log_prob(sampled_actions)
        student_logp = student_dist.log_prob(sampled_actions)

        if teacher_logp.dim() > 1:
            teacher_logp = teacher_logp.sum(dim=-1)
        if student_logp.dim() > 1:
            student_logp = student_logp.sum(dim=-1)

        kl = (teacher_logp - student_logp).mean()
        distil_loss = kl * self.weight

        model.policy.optimizer.zero_grad()
        distil_loss.backward()
        model.policy.optimizer.step()

        if self.verbose:
            print(f"[Distill] loss {distil_loss.item():.6f}")
        return True

# -----------------------------
# Training pipeline
# -----------------------------
def train_low_level():
    print("=== Train low-level ===")
    base_ctor = lambda: gym.make(ENV_ID)
    def make_ll(rank):
        def _init():
            e = LowLevelEnv(base_ctor, command_dim=1, command_low=0.0, command_high=3.0)
            # ensure deterministic seed for subprocess
            e.reset(seed=SEED + rank)
            return e
        return _init

    envs = SubprocVecEnv([make_ll(i) for i in range(NUM_ENVS)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.)

    eval_env = LowLevelEnv(base_ctor, command_dim=1)
    low_agent = SAC("MlpPolicy", envs, verbose=1, device=DEVICE, batch_size=BATCH_SIZE,
                    learning_rate=3e-4, buffer_size=1_000_000)

    eval_cb = EvaluationCallback(eval_env, eval_freq=50_000, n_eval_episodes=3, verbose=1)
    low_agent.learn(total_timesteps=TOTAL_TIMESTEPS_LOW, callback=eval_cb, reset_num_timesteps=True)
    low_agent.save(os.path.join(LOG_DIR, "low_agent"))
    envs.close()
    eval_env.close()
    return low_agent

def train_high_level(low_agent: SAC, teacher_model: Optional[SAC] = None):
    print("=== Train high-level ===")
    base_ctor = lambda: gym.make(ENV_ID)
    def make_hl(rank):
        def _init():
            e = HighLevelEnv(base_ctor, low_agent=low_agent, command_dim=1, low_level_duration=LOW_LEVEL_DURATION)
            e.reset(seed=SEED + rank)
            return e
        return _init

    envs = SubprocVecEnv([make_hl(i) for i in range(NUM_ENVS)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True)
    eval_env = HighLevelEnv(base_ctor, low_agent=low_agent, command_dim=1)

    high_agent = SAC("MlpPolicy", envs, verbose=1, device=DEVICE, batch_size=BATCH_SIZE,
                     learning_rate=3e-4, buffer_size=1_000_000)

    callbacks = [EvaluationCallback(eval_env, eval_freq=50_000, n_eval_episodes=3, verbose=1)]
    if teacher_model is not None:
        teacher_model.policy.to(DEVICE)
        callbacks.append(DistillationCallback(teacher_model, distill_weight=DISTILLATION_WEIGHT, every_n_calls=500, verbose=1))

    high_agent.learn(total_timesteps=TOTAL_TIMESTEPS_HIGH, callback=callbacks, reset_num_timesteps=True)
    high_agent.save(os.path.join(LOG_DIR, "high_agent"))
    envs.close()
    eval_env.close()
    return high_agent

def train_single_sac():
    print("=== Train Single SAC Agent ===")
    
    # 1. Environment Setup
    # 기본 환경 생성 함수
    base_ctor = lambda: gym.make(ENV_ID)
    
    # 단일 에이전트 학습을 위한 환경 팩토리 (MaxSpeedRewardWrapper 적용)
    def make_single_env(rank):
        def _init():
            env = MaxSpeedRewardWrapper(base_ctor())
            env.reset(seed=SEED + rank)
            return env
        return _init

    # SubprocVecEnv와 VecNormalize를 사용하여 병렬 학습 환경 구성
    envs = SubprocVecEnv([make_single_env(i) for i in range(NUM_ENVS)])
    # Low-LevelEnv와 달리 Command가 없으므로 VecNormalize의 norm_obs는 유지
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 2. Evaluation Environment Setup
    eval_env_base = base_ctor()
    eval_env = MaxSpeedRewardWrapper(eval_env_base)
    
    # 3. Model Initialization
    single_agent = SAC("MlpPolicy", envs, verbose=1, device=DEVICE, batch_size=BATCH_SIZE,
                       learning_rate=3e-4, buffer_size=1_000_000)

    # 4. Callback for Evaluation
    # Low-Level 대신 Single Agent에 맞게 평가 빈도 조정 가능
    eval_cb = EvaluationCallback(eval_env, eval_freq=50_000, n_eval_episodes=5, verbose=1)
    
    # 5. Training
    # Low-level과 High-level의 총 스텝을 합쳐서 단일 에이전트에 부여
    total_timesteps_single = TOTAL_TIMESTEPS_LOW + TOTAL_TIMESTEPS_HIGH  # 예: 1.5M
    single_agent.learn(total_timesteps=total_timesteps_single, callback=eval_cb, reset_num_timesteps=True)
    
    single_agent.save(os.path.join(LOG_DIR, "single_sac_agent"))
    envs.close()
    eval_env.close()
    print("Single SAC Training Done.")
    return single_agent


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    set_random_seed(SEED)

    # Low-Level / High-Level HRL 대신 단일 SAC 학습을 실행합니다.
    single_agent = train_single_sac()

    print("Done. Model saved in", LOG_DIR)

# if __name__ == "__main__":
#     set_random_seed(SEED)

#     # low_path = os.path.join(LOG_DIR, "low_agent.zip")
#     # if os.path.exists(low_path):
#     #     print("Loading low-level agent...")
#     #     low_agent = SAC.load(low_path, device=DEVICE)
#     # else:
#     low_agent = train_low_level()

#     teacher_model = None
#     if TEACHER_PATH and os.path.exists(TEACHER_PATH):
#         teacher_model = SAC.load(TEACHER_PATH, device=DEVICE)
#     else:
#         print("No teacher provided.")

#     high_path = os.path.join(LOG_DIR, "high_agent.zip")
#     if os.path.exists(high_path):
#         print("Loading high-level agent...")
#         high_agent = SAC.load(high_path, device=DEVICE)
#     else:
#         high_agent = train_high_level(low_agent=low_agent, teacher_model=teacher_model)

#     print("Done. Models saved in", LOG_DIR)
