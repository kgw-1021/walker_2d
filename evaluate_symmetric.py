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

# Environment setup
# Note: These lines are for local environment setup and may not be necessary in all execution environments.
# os.environ['MUJOCO_GL'] = 'glfw'
# os.environ['SDL_VIDEODRIVER'] = 'windows'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==================== Config & Trajectory Generator ====================
class Config:
    env_name = "Walker2d-v5"
    
    # Adaptive Gait Parameters
    stride_length_base = 0.35
    stride_length_coef = 0.25
    stride_time_base = 0.8
    stride_time_coef = -0.15
    x_vel_min = 0.1
    
    # Target Trajectory Parameters (Sinusoidal pattern)
    hip_amplitude = 0.25
    hip_center = 0.0
    hip_offset = 0.0
    knee_amplitude = 0.5
    knee_center = 0.8
    knee_offset = -np.pi/2
    
    # Reward Weights (Final values for evaluation)
    lambda_vel_final = 1.0       
    lambda_track_final = 1.0   
    lambda_ctrl = 0.0005
    
    log_dir = "./walker_phase_sac_logs/"
    
    # Evaluation Settings
    n_eval_episodes = 10
    max_test_steps = 1000
    
    # Video Settings
    save_video = False
    video_folder = "./walker_phase_sac_logs/videos/"
    video_name_prefix = "walker2d_evaluation"

class TrajectoryGenerator:
    def __init__(self, config):
        self.cfg = config
    
    def get_target_angles(self, phase):
        # Left Leg Target
        left_hip = self.cfg.hip_amplitude * np.sin(2 * np.pi * phase + self.cfg.hip_offset) + self.cfg.hip_center
        left_knee = self.cfg.knee_amplitude * np.sin(2 * np.pi * phase + self.cfg.knee_offset) + self.cfg.knee_center
        
        # Right Leg Target (180 degree phase difference for symmetric gait)
        right_phase = (phase + 0.5) % 1.0
        right_hip = self.cfg.hip_amplitude * np.sin(2 * np.pi * right_phase + self.cfg.hip_offset) + self.cfg.hip_center
        right_knee = self.cfg.knee_amplitude * np.sin(2 * np.pi * right_phase + self.cfg.knee_offset) + self.cfg.knee_center
        
        return {
            'left_hip': left_hip, 'left_knee': left_knee,
            'right_hip': right_hip, 'right_knee': right_knee
        }

# ==================== Phase-Augmented Wrapper (Trajectory Recording Added) ====================
class PhaseAugmentedWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.cfg = config
        self.traj_gen = TrajectoryGenerator(config)
        # Walker2d joint indices: (rootx, rooty, rootz, hip_left, knee_left, hip_right, knee_right, ankle_left, ankle_right)
        self.joint_indices = {'left_hip': 3, 'left_knee': 4, 'right_hip': 6, 'right_knee': 7}
        
        self.phase = 0.0
        self.time = 0.0
        self.dt = 0.008
        self.current_vel_weight = self.cfg.lambda_vel_final
        self.current_track_weight = self.cfg.lambda_track_final
        
        # Evaluation variables
        self.episode_energy = 0.0
        self.episode_distance = 0.0
        self.initial_x = 0.0
        
        # Trajectory recording variables
        self.phase_history = []
        self.actual_trajectories = {k: [] for k in self.joint_indices}
        self.target_trajectories = {k: [] for k in self.joint_indices}
        
        # Observation space augmentation
        base_dim = self.env.observation_space.shape[0]
        new_dim = base_dim + 6 # +2 for phase, +4 for errors
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
        
        # Reset trajectory history
        self.phase_history = []
        self.actual_trajectories = {k: [] for k in self.joint_indices}
        self.target_trajectories = {k: [] for k in self.joint_indices}
        
        return self._augment_observation(obs), info
        
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        x_vel = info.get('x_velocity', self.env.unwrapped.data.qvel[0])
        qpos = self.env.unwrapped.data.qpos
        
        # Adaptive parameters
        stride_length, stride_time = self._get_adaptive_stride_params(x_vel)
        
        self.time += self.dt
        delta_phi = self.dt / stride_time
        self.phase = (self.phase + delta_phi) % 1.0
        
        aug_obs = self._augment_observation(obs)
        custom_reward, tracking_error_L1, targets = self._compute_reward_and_targets(action, info) 
        
        # ===== Trajectory Recording =====
        self.phase_history.append(self.phase)
        for name, idx in self.joint_indices.items():
            # qpos is the full joint position vector
            self.actual_trajectories[name].append(qpos[idx])
            self.target_trajectories[name].append(targets[name])
        
        # ===== Evaluation Metrics Calculation =====
        qvel = self.env.unwrapped.data.qvel
        
        # 1. Energy consumption (for CoT)
        # Note: Walker2d has 6 active joints starting at index 3 (hip_left, knee_left, ankle_left, hip_right, knee_right, ankle_right)
        energy_step = np.sum(np.abs(action * qvel[3:9]))
        self.episode_energy += energy_step * self.dt
        
        # 2. Distance traveled
        current_x = qpos[0]
        self.episode_distance = current_x - self.initial_x
        
        # 3. Symmetry Error
        curr = {name: qpos[idx] for name, idx in self.joint_indices.items()}
        
        left_error = abs(curr['left_hip'] - targets['left_hip']) + abs(curr['left_knee'] - targets['left_knee'])
        right_error = abs(curr['right_hip'] - targets['right_hip']) + abs(curr['right_knee'] - targets['right_knee'])
        symmetry_error = abs(left_error - right_error)
        
        # 4. Torso Stability (rooty angle)
        torso_angle = abs(qpos[2])
        
        # Add metrics to Info
        info['energy_step'] = energy_step
        info['total_energy'] = self.episode_energy
        info['distance'] = self.episode_distance
        info['symmetry_error'] = symmetry_error
        info['torso_angle'] = torso_angle
        info['x_velocity'] = x_vel
        info['tracking_error_L1'] = tracking_error_L1
        info['stride_length'] = stride_length
        info['stride_time'] = stride_time
        
        # Return trajectory data upon termination/truncation for plotting
        if terminated or truncated:
            info['episode_trajectories'] = {
                'phase': np.array(self.phase_history),
                'actual': {k: np.array(v) for k, v in self.actual_trajectories.items()},
                'target': {k: np.array(v) for k, v in self.target_trajectories.items()}
            }
        
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

    def _compute_reward_and_targets(self, action, info):
        x_vel = info.get('x_velocity', self.env.unwrapped.data.qvel[0])
        qpos = self.env.unwrapped.data.qpos
        
        curr = {name: qpos[idx] for name, idx in self.joint_indices.items()}
        targets = self.traj_gen.get_target_angles(self.phase) # Calculate targets
        
        tracking_error_L1 = sum([abs(curr[k] - targets[k]) for k in self.joint_indices])
        control_cost = np.sum(np.square(action))
        healthy_reward = 5.0
        
        reward = (
            self.current_vel_weight * x_vel
            + healthy_reward
            - self.current_track_weight * tracking_error_L1
            - self.cfg.lambda_ctrl * control_cost
        )
        return reward, tracking_error_L1, targets # Return targets

def make_eval_env(config, render_mode=None, save_video=False):
    env = gym.make(config.env_name, render_mode=render_mode)
    env = PhaseAugmentedWrapper(env, config)
    
    # Add video recording wrapper
    if save_video:
        os.makedirs(config.video_folder, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=config.video_folder,
            episode_trigger=lambda x: True,
            name_prefix=config.video_name_prefix
        )
        print(f"✓ Video saving configured: {config.video_folder}")
    
    return Monitor(env)

# ==================== Trajectory Plotting Function (English & Improved Readability) ====================

def plot_joint_trajectories(trajectories, config, episode_index):
    """
    Joint trajectories (Target vs. Actual) visualization. (English for compatibility)
    """
    phase = trajectories['phase']
    actual = trajectories['actual']
    target = trajectories['target']
    
    # Joint names
    joint_names = ['left_hip', 'left_knee', 'right_hip', 'right_knee']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Episode {episode_index} - Joint Trajectory Tracking Analysis', fontsize=16, fontweight='bold')
    
    # 2x2 subplot configuration
    ax_map = {
        'left_hip': axes[0, 0], 'left_knee': axes[0, 1],
        'right_hip': axes[1, 0], 'right_knee': axes[1, 1]
    }
    
    for name in joint_names:
        ax = ax_map[name]
        
        # 1. Target Trajectory
        ax.plot(phase, target[name], label='Target Trajectory', color='red', linestyle='--', linewidth=2.0)
        
        # 2. Actual Trajectory 
        ax.plot(phase, actual[name], label='Actual Trajectory', color='blue', alpha=0.7, linewidth=1.5)
        
        # 3. Calculate Error
        error = np.abs(actual[name] - target[name])
        mean_error = np.mean(error)
        
        # 4. Set Title and Labels (English)
        title_name = name.replace("_", " ").title()
        ax.set_title(f'{title_name} (Mean L1 Error: {mean_error:.3f} rad)')
        ax.set_xlabel('Gait Phase ($\phi$, 0.0 to 1.0)')
        ax.set_ylabel('Joint Angle (rad)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 5. Y-axis Scale Adjustment for better focus on movement
        if 'hip' in name:
            # Hip: Centered around 0.0 with amplitude of 0.25 (range approx -0.5 to 0.5)
            ax.set_ylim([-1, 1]) 
        elif 'knee' in name:
            # Knee: Center 0.8, amplitude 0.5 (range approx 0.3 to 1.3 rad, but actual movement can be lower)
            ax.set_ylim([-2, 5]) 

        ax.set_xlim([0, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(config.log_dir, f'trajectories_ep_{episode_index}_{datetime.now().strftime("%H%M%S")}.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 Trajectory plot saved (Ep {episode_index}): {save_path}")
    plt.show()

# ==================== Metrics Plotting Function (English) ====================

def plot_metrics(metrics, config):
    """Evaluation Metrics Visualization (Labels in English)"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Walker2D Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # 1. Velocity
    axes[0, 0].bar(range(len(metrics['velocities'])), metrics['velocities'], color='skyblue')
    axes[0, 0].axhline(np.mean(metrics['velocities']), color='red', linestyle='--', label='Mean')
    axes[0, 0].set_title('Average Velocity (m/s)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].legend()
    
    # 2. Symmetry Error
    axes[0, 1].bar(range(len(metrics['symmetry_errors'])), metrics['symmetry_errors'], color='lightcoral')
    axes[0, 1].axhline(np.mean(metrics['symmetry_errors']), color='red', linestyle='--', label='Mean')
    axes[0, 1].set_title('Symmetry Error (rad, Lower is Better)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].legend()
    
    # 3. CoT
    valid_cots = [c if c != float('inf') else 0 for c in metrics['cots']]
    axes[0, 2].bar(range(len(valid_cots)), valid_cots, color='lightgreen')
    axes[0, 2].set_title('Cost of Transport (Lower is Better)')
    axes[0, 2].set_xlabel('Episode')
    
    # 4. Stride Length
    axes[1, 0].bar(range(len(metrics['stride_lengths'])), metrics['stride_lengths'], color='orange')
    axes[1, 0].axhline(np.mean(metrics['stride_lengths']), color='red', linestyle='--')
    axes[1, 0].set_title('Average Stride Length (m)')
    axes[1, 0].set_xlabel('Episode')
    
    # 5. Torso Angle (Stability)
    axes[1, 1].bar(range(len(metrics['torso_angles'])), metrics['torso_angles'], color='plum')
    axes[1, 1].axhline(np.mean(metrics['torso_angles']), color='red', linestyle='--')
    axes[1, 1].set_title('Torso Angle (rad, Closer to 0 is Stable)')
    axes[1, 1].set_xlabel('Episode')
    
    # 6. Survival Rate
    axes[1, 2].bar(range(len(metrics['survival_rates'])), metrics['survival_rates'], color='mediumpurple')
    axes[1, 2].axhline(np.mean(metrics['survival_rates']), color='red', linestyle='--')
    axes[1, 2].set_title('Survival Rate (Higher is Better)')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylim([0, 1.1])
    
    plt.tight_layout()
    save_path = os.path.join(config.log_dir, 'evaluation_metrics.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n📊 Metrics plot saved: {save_path}")
    plt.show()


# ==================== Evaluation Function (Tracking Error & Plotting Integration) ====================

def evaluate_with_advanced_metrics(config, render=False, save_video=False):
    """Evaluation with advanced metrics including Tracking Error and Trajectory Plotting."""
    
    # 1. Load Environment and Model
    stats_path = os.path.join(config.log_dir, "vec_normalize.pkl")
    model_path = os.path.join(config.log_dir, "best_model", "best_model.zip")
    
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        print("❌ ERROR: Model or statistics file not found.")
        return

    render_mode = "rgb_array" if save_video else ("human" if render else None)
    
    eval_env = DummyVecEnv([lambda: make_eval_env(config, render_mode=render_mode, save_video=save_video)])
    eval_env = VecNormalize.load(stats_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False
    
    model = SAC.load(model_path)
    print(f"✅ Model loaded: {model_path}")
    
    if save_video:
        print(f"🎥 Video recording mode active\n")

    # 2. Initialize Metrics
    metrics = {
        'rewards': [], 'velocities': [], 'distances': [], 'energies': [],
        'cots': [], 'symmetry_errors': [], 'torso_angles': [],
        'survival_rates': [], 'stride_lengths': [], 'stride_times': [],
        'tracking_errors': [] # Tracking Error added
    }
    
    print(f"📊 Starting evaluation for {config.n_eval_episodes} episodes...\n")
    
    for episode in range(config.n_eval_episodes):
        obs = eval_env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        # Per-episode metrics lists
        ep_velocities = []
        ep_symmetry_errors = []
        ep_torso_angles = []
        ep_stride_lengths = []
        ep_stride_times = []
        ep_tracking_errors = [] 
        trajectories_data = None 
        
        while not done and step_count < config.max_test_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            episode_reward += reward[0]
            step_count += 1
            
            info_data = info[0]
            
            # Real-time metrics collection
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
            if 'tracking_error_L1' in info_data: 
                ep_tracking_errors.append(info_data['tracking_error_L1'])
            
            # Collect trajectory data (if available on termination/truncation)
            if 'episode_trajectories' in info_data:
                trajectories_data = info_data['episode_trajectories']
                
            done = done[0]

        # Post-episode metrics calculation
        if step_count > 0:
            final_info = info[0]
            
            avg_velocity = np.mean(ep_velocities) if ep_velocities else 0
            distance = final_info.get('distance', 0)
            total_energy = final_info.get('total_energy', 0)
            
            # Cost of Transport (CoT)
            mass = 3.5
            cot = total_energy / (mass * distance) if distance > 0 else float('inf')
            
            avg_symmetry = np.mean(ep_symmetry_errors) if ep_symmetry_errors else 0
            avg_torso_angle = np.mean(ep_torso_angles) if ep_torso_angles else 0
            survival_rate = step_count / config.max_test_steps
            avg_stride_length = np.mean(ep_stride_lengths) if ep_stride_lengths else 0
            avg_stride_time = np.mean(ep_stride_times) if ep_stride_times else 0
            avg_tracking_error = np.mean(ep_tracking_errors) if ep_tracking_errors else 0 
            
            # Save
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
            metrics['tracking_errors'].append(avg_tracking_error)
            
            # Episode-wise Output (English/Clarity)
            print(f"[EP {episode+1:2d}] "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Vel: {avg_velocity:.2f}m/s | "
                  f"Symm: {avg_symmetry:.3f} | "
                  f"Track Err: {avg_tracking_error:.3f} | " 
                  f"Survival: {survival_rate:.1%}")
                  
            # Plot Trajectories
            if trajectories_data is not None:
                plot_joint_trajectories(trajectories_data, config, episode + 1)


    eval_env.close()
    
    if save_video:
        print(f"\n  Video saving complete: {config.video_folder}")
    
    # 3. Final Statistics Output (English)
    print("\n" + "="*70)
    print("                        Final Evaluation Results")
    print("="*70)
    
    print(f"\n  Performance Metrics (Avg over {config.n_eval_episodes} Episodes)")
    print("-"*70)
    print(f"  1. Total Reward:                     {np.mean(metrics['rewards']):8.2f} ± {np.std(metrics['rewards']):.2f}")
    print(f"  2. Avg Velocity:                     {np.mean(metrics['velocities']):8.2f} m/s ± {np.std(metrics['velocities']):.2f}")
    print(f"  3. Total Distance:                   {np.mean(metrics['distances']):8.2f} m ± {np.std(metrics['distances']):.2f}")
    print(f"  4. Survival Rate:                    {np.mean(metrics['survival_rates']):8.1%} ± {np.std(metrics['survival_rates']):.1%}")
    
    print(f"\n  Gait Quality Metrics")
    print("-"*70)
    print(f"  5. Avg Tracking Error (L1):          {np.mean(metrics['tracking_errors']):8.4f} rad ± {np.std(metrics['tracking_errors']):.4f}")
    print(f"     (Lower is better, overall deviation from target trajectory)")
    print(f"  6. Symmetry Error:                   {np.mean(metrics['symmetry_errors']):8.4f} rad ± {np.std(metrics['symmetry_errors']):.4f}")
    print(f"     (Closer to 0 is better, consistency between left/right tracking)")
    print(f"  7. Torso Stability (Torso Angle):    {np.mean(metrics['torso_angles']):8.4f} rad ± {np.std(metrics['torso_angles']):.4f}")
    
    print(f"\n Efficiency Metrics")
    print("-"*70)
    valid_cots = [c for c in metrics['cots'] if c != float('inf')]
    if valid_cots:
        print(f"  8. CoT (Cost of Transport):          {np.mean(valid_cots):8.3f} ± {np.std(valid_cots):.3f}")
        print(f"  9. Total Energy Consumed:            {np.mean(metrics['energies']):8.2f} J ± {np.std(metrics['energies']):.2f}")
    
    print(f"\n  Gait Parameters")
    print("-"*70)
    print(f" 10. Avg Stride Length:               {np.mean(metrics['stride_lengths']):8.3f} m ± {np.std(metrics['stride_lengths']):.3f}")
    print(f" 11. Avg Stride Time:                 {np.mean(metrics['stride_times']):8.3f} s ± {np.std(metrics['stride_times']):.3f}")
    
    # 4. Overall Gait Quality Score Calculation (Tracking Score Added)
    print(f"\n🌟 Overall Gait Quality Score")
    print("-"*70)
    
    # Normalized Scores (0-100)
    velocity_score = min(np.mean(metrics['velocities']) / 1.5 * 100, 100)  # Target 1.5 m/s
    tracking_score = max(0, 100 - np.mean(metrics['tracking_errors']) * 150) # Target low error
    symmetry_score = max(0, 100 - np.mean(metrics['symmetry_errors']) * 100)
    stability_score = max(0, 100 - np.mean(metrics['torso_angles']) * 200)
    survival_score = np.mean(metrics['survival_rates']) * 100
    efficiency_score = max(0, 100 - np.mean(valid_cots) * 10) if valid_cots else 0
    
    # Adjusted weights (Tracking Score added)
    overall_score = (
        velocity_score * 0.20 + 
        tracking_score * 0.20 + 
        symmetry_score * 0.20 + 
        stability_score * 0.20 + 
        efficiency_score * 0.10 +
        survival_score * 0.10
    )
    
    print(f"  - Velocity Score:      {velocity_score:5.1f}/100")
    print(f"  - **Tracking Score**:  {tracking_score:5.1f}/100 (Accuracy of Trajectory)")
    print(f"  - Symmetry Score:      {symmetry_score:5.1f}/100")
    print(f"  - Stability Score:     {stability_score:5.1f}/100")
    print(f"  - Efficiency Score:    {efficiency_score:5.1f}/100")
    print(f"  - Survival Score:      {survival_score:5.1f}/100")
    print(f"\n  ⭐ FINAL SCORE:         {overall_score:5.1f}/100")
    
    print("="*70)
    
    # 5. Plot Metrics
    plot_metrics(metrics, config)
    
    return metrics

if __name__ == "__main__":
    config = Config()
    save_video = config.save_video
    metrics = evaluate_with_advanced_metrics(config, render=False, save_video=save_video)