#!/usr/bin/env python3
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

class DataVisualizer:
    def __init__(self):
        # Define joint names for Go1 robot
        self.joint_names = [
            'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip',    # Hip joints
            'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh',  # Thigh joints
            'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf'   # Calf joints
        ]
        
        # Define observation components for 42-dimensional observation
        self.obs_components = {
            'projected_gravity': {'start': 0, 'size': 3, 'labels': ['gravity_x', 'gravity_y', 'gravity_z']},
            'joint_pos': {'start': 3, 'size': 12, 'labels': self.joint_names},
            'joint_vel': {'start': 15, 'size': 12, 'labels': [f"{name}_vel" for name in self.joint_names]},
            'last_action': {'start': 27, 'size': 12, 'labels': [f"{name}_action" for name in self.joint_names]},
            'velocity_commands': {'start': 39, 'size': 3, 'labels': ['cmd_x', 'cmd_y', 'cmd_z']}
        }

    def load_isaac_data(self, pkl_path, log_dir_isaac="visualization_results_0712_2"):
        """Load data recorded from Isaac Sim (record.py)"""
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            actions = data['actions']
            observations = data['observations']
            
            # Remove batch dimension if present
            if len(actions.shape) == 3:
                actions = actions[:, 0, :]  # Take first environment
            if len(observations.shape) == 3:
                observations = observations[:, 0, :]  # Take first environment
                
            print(f"Isaac data loaded: {actions.shape[0]} steps")
            print(f"Action shape: {actions.shape}, Observation shape: {observations.shape}")

            # Check if log_dir_isaac exists
            if not os.path.exists(log_dir_isaac):
                print(f"Warning: Isaac log directory not found: {log_dir_isaac}")
                processed_actions = np.zeros_like(actions)
            else:
                # Look for both .pt and .npy files
                processed_action_files_pt = sorted([f for f in os.listdir(log_dir_isaac) if f.startswith('processed_actions') and f.endswith('.pt')])
                processed_action_files_npy = sorted([f for f in os.listdir(log_dir_isaac) if f.startswith('processed_actions_20250715_034042') and f.endswith('.npy')])
                
                all_processed_actions = []
                
                # Load .pt files
                for file in processed_action_files_pt:
                    try:
                        processed_action_data = torch.load(os.path.join(log_dir_isaac, file))
                        processed_action_numpy = processed_action_data.numpy()
                        
                        # Remove extra dimensions if present
                        if len(processed_action_numpy.shape) == 3 and processed_action_numpy.shape[1] == 1:
                            processed_action_numpy = processed_action_numpy.squeeze(1)
                        
                        all_processed_actions.append(processed_action_numpy)
                        print(f"Loaded processed action file: {file}, shape: {processed_action_numpy.shape}")
                    except Exception as e:
                        print(f"Error loading .pt file {file}: {e}")
                
                # Load .npy files
                for file in processed_action_files_npy:
                    try:
                        processed_action_numpy = np.load(os.path.join(log_dir_isaac, file))
                        
                        # Remove extra dimensions if present
                        if len(processed_action_numpy.shape) == 3 and processed_action_numpy.shape[1] == 1:
                            processed_action_numpy = processed_action_numpy.squeeze(1)
                        
                        all_processed_actions.append(processed_action_numpy)
                        print(f"Loaded processed action file: {file}, shape: {processed_action_numpy.shape}")
                    except Exception as e:
                        print(f"Error loading .npy file {file}: {e}")
                
                if all_processed_actions:
                    processed_actions = np.concatenate(all_processed_actions, axis=0)
                    print(f"Total processed actions shape: {processed_actions.shape}")
                else:
                    print("Warning: No processed action files found, using zeros")
                    processed_actions = np.zeros_like(actions)

            return {
                'actions': actions,
                'observations': observations,
                'processed_actions': processed_actions,
                'timesteps': np.arange(len(actions))
            }
        
        except Exception as e:
                print(f"Error loading Isaac data: {e}")
                return None

    def load_mujoco_robot_data(self, log_dir):
        """Load data recorded from mujoco (go1_rl_mujoco_insert_actions.py)"""
        try:
            # Find all observation and action files
            obs_files = sorted([f for f in os.listdir(log_dir) if f.startswith('observations_step_') and f.endswith('.pt')])
            action_files = sorted([f for f in os.listdir(log_dir) if f.startswith('actions_step_') and f.endswith('.pt')])
            processed_action_files = sorted([f for f in os.listdir(log_dir) if f.startswith('processed_actions_step_') and f.endswith('.pt')])
            
            if not obs_files or not action_files or not processed_action_files:
                print(f"No data files found in {log_dir}")
                return None
            
            # Load and concatenate all data
            all_observations = []
            all_actions = []
            all_processed_actions = []
            
            for obs_file, action_file, processed_action_file in zip(obs_files, action_files, processed_action_files):
                obs_data = torch.load(os.path.join(log_dir, obs_file))
                action_data = torch.load(os.path.join(log_dir, action_file))
                processed_action_data = torch.load(os.path.join(log_dir, processed_action_file))
                
                # Convert to numpy and handle shape issues
                obs_numpy = obs_data.numpy()
                action_numpy = action_data.numpy()
                processed_action_numpy = processed_action_data.numpy()
                
                # Remove extra dimensions if present
                if len(obs_numpy.shape) == 3 and obs_numpy.shape[1] == 1:
                    obs_numpy = obs_numpy.squeeze(1)  # Remove the middle dimension (1000, 1, 42) -> (1000, 42)
                if len(action_numpy.shape) == 3 and action_numpy.shape[1] == 1:
                    action_numpy = action_numpy.squeeze(1)  # Remove extra dimension if present
                if len(processed_action_numpy.shape) == 3 and processed_action_numpy.shape[1] == 1:
                    processed_action_numpy = processed_action_numpy.squeeze(1)  # Remove extra dimension if present
                
                all_observations.append(obs_numpy)
                all_actions.append(action_numpy)
                all_processed_actions.append(processed_action_numpy)
            
            observations = np.concatenate(all_observations, axis=0)
            actions = np.concatenate(all_actions, axis=0)
            processed_actions = np.concatenate(all_processed_actions, axis=0)
            
            print(f"mujoco data loaded: {actions.shape[0]} steps")
            print(f"Action shape: {actions.shape}, Observation shape: {observations.shape}")
            print(f"Processed Action shape: {processed_actions.shape}")
            
            return {
                'actions': actions,
                'observations': observations,
                'processed_actions': processed_actions,
                'timesteps': np.arange(len(actions))
            }
        except Exception as e:
            print(f"Error loading mujoco data: {e}")
            return None

    def plot_actions(self, isaac_data, mujoco_data, save_dir):
        """Plot action comparisons"""
        if isaac_data is None or mujoco_data is None:
            print("Cannot plot actions: missing data")
            return
        
        # Determine the minimum length for comparison
        min_len = min(len(isaac_data['actions']), len(mujoco_data['actions']))
        
        # Create subplots for each joint
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('Joint Actions Comparison: Isaac Sim vs mujoco', fontsize=16)
        
        for i, joint_name in enumerate(self.joint_names):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Plot Isaac data
            isaac_actions = isaac_data['actions'][:min_len, i]
            mujoco_actions = mujoco_data['actions'][:min_len, i]
            
            timesteps = np.arange(min_len)
            
            ax.plot(timesteps, isaac_actions, label='Isaac Sim', alpha=0.7, linewidth=1)
            ax.plot(timesteps, mujoco_actions, label='mujoco', alpha=0.7, linewidth=1)
            
            ax.set_title(f'{joint_name}')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Action Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'actions_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Actions comparison plot saved")

    def plot_processed_actions(self, isaac_data, mujoco_data, save_dir):
        """Plot action comparisons"""
        if isaac_data is None or mujoco_data is None:
            print("Cannot plot actions: missing data")
            return
        
        # Determine the minimum length for comparison
        min_len = min(len(isaac_data['actions']), len(mujoco_data['actions']))
        
        # Create subplots for each joint
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('Processed Joint Actions Comparison: Isaac Sim vs mujoco', fontsize=16)
        
        for i, joint_name in enumerate(self.joint_names):
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Plot Isaac data
            isaac_actions = isaac_data['processed_actions'][:min_len, i]
            mujoco_actions = mujoco_data['processed_actions'][:min_len, i]
            
            timesteps = np.arange(min_len)
            
            ax.plot(timesteps, isaac_actions, label='Isaac Sim', alpha=0.7, linewidth=1)
            ax.plot(timesteps, mujoco_actions, label='mujoco', alpha=0.7, linewidth=1)
            
            ax.set_title(f'{joint_name}')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Action Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'processed_actions_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Actions comparison plot saved")

    def plot_observations(self, isaac_data, mujoco_data, save_dir):
        """Plot observation comparisons for each component"""
        if isaac_data is None or mujoco_data is None:
            print("Cannot plot observations: missing data")
            return
        
        min_len = min(len(isaac_data['observations']), len(mujoco_data['observations']))
        timesteps = np.arange(min_len)
        
        # Verify both datasets have 42-dimensional observations
        isaac_obs_dim = isaac_data['observations'].shape[1]
        mujoco_obs_dim = mujoco_data['observations'].shape[1]
        
        print(f"Isaac observation dimension: {isaac_obs_dim}")
        print(f"mujoco observation dimension: {mujoco_obs_dim}")
        
        if isaac_obs_dim != 42 or mujoco_obs_dim != 42:
            print(f"Warning: Expected 42-dimensional observations, got Isaac: {isaac_obs_dim}, mujoco: {mujoco_obs_dim}")
            return
        
        for component_name, component_info in self.obs_components.items():
            start_idx = component_info['start']
            size = component_info['size']
            labels = component_info['labels']
            
            # Create subplots for this component
            if size <= 3:
                fig, axes = plt.subplots(1, size, figsize=(5*size, 4))
                if size == 1:
                    axes = [axes]
            elif size <= 6:
                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes = axes.flatten()
            else:
                fig, axes = plt.subplots(4, 3, figsize=(15, 12))
                axes = axes.flatten()
            
            fig.suptitle(f'{component_name.replace("_", " ").title()} Comparison: Isaac Sim vs mujoco', fontsize=16)
            
            for i in range(size):
                if i >= len(axes):
                    break
                    
                isaac_obs = isaac_data['observations'][:min_len, start_idx + i]
                mujoco_obs = mujoco_data['observations'][:min_len, start_idx + i]
                
                axes[i].plot(timesteps, isaac_obs, label='Isaac Sim', alpha=0.7, linewidth=1)
                axes[i].plot(timesteps, mujoco_obs, label='mujoco', alpha=0.7, linewidth=1)
                
                axes[i].set_title(f'{labels[i]}')
                axes[i].set_xlabel('Timestep')
                axes[i].set_ylabel('Value')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(size, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{component_name}_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{component_name} comparison plot saved")

    def plot_statistical_comparison(self, isaac_data, mujoco_data, save_dir):
        """Plot statistical comparison (mean, std, etc.)"""
        if isaac_data is None or mujoco_data is None:
            print("Cannot plot statistical comparison: missing data")
            return
        
        min_len = min(len(isaac_data['actions']), len(mujoco_data['actions']))
        
        # Action statistics
        isaac_actions = isaac_data['actions'][:min_len]
        mujoco_actions = mujoco_data['actions'][:min_len]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean comparison
        isaac_mean = np.mean(isaac_actions, axis=0)
        mujoco_mean = np.mean(mujoco_actions, axis=0)
        
        x = np.arange(len(self.joint_names))
        width = 0.35
        
        ax1.bar(x - width/2, isaac_mean, width, label='Isaac Sim', alpha=0.7)
        ax1.bar(x + width/2, mujoco_mean, width, label='mujoco', alpha=0.7)
        ax1.set_xlabel('Joint')
        ax1.set_ylabel('Mean Action Value')
        ax1.set_title('Mean Action Values Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.joint_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Standard deviation comparison
        isaac_std = np.std(isaac_actions, axis=0)
        mujoco_std = np.std(mujoco_actions, axis=0)
        
        ax2.bar(x - width/2, isaac_std, width, label='Isaac Sim', alpha=0.7)
        ax2.bar(x + width/2, mujoco_std, width, label='mujoco', alpha=0.7)
        ax2.set_xlabel('Joint')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Action Standard Deviation Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.joint_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'statistical_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Statistical comparison plot saved")

    def plot_detailed_joint_comparison(self, isaac_data, mujoco_data, save_dir):
        """Plot detailed comparison for each joint separately"""
        if isaac_data is None or mujoco_data is None:
            return
        
        min_len = min(len(isaac_data['actions']), len(mujoco_data['actions']))
        timesteps = np.arange(min_len)
        
        # Create individual plots for each joint
        for i, joint_name in enumerate(self.joint_names):
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(f'{joint_name} Detailed Comparison', fontsize=16)
            
            # Action comparison
            isaac_actions = isaac_data['actions'][:min_len, i]
            mujoco_actions = mujoco_data['actions'][:min_len, i]
            
            ax1.plot(timesteps, isaac_actions, label='Isaac Sim', alpha=0.7, linewidth=1)
            ax1.plot(timesteps, mujoco_actions, label='mujoco', alpha=0.7, linewidth=1)
            ax1.set_title('Actions')
            ax1.set_ylabel('Action Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Joint position comparison (from observations)
            isaac_joint_pos = isaac_data['observations'][:min_len, 3 + i]  # joint_pos starts at index 3
            mujoco_joint_pos = mujoco_data['observations'][:min_len, 3 + i]
            
            ax2.plot(timesteps, isaac_joint_pos, label='Isaac Sim', alpha=0.7, linewidth=1)
            ax2.plot(timesteps, mujoco_joint_pos, label='mujoco', alpha=0.7, linewidth=1)
            ax2.set_title('Joint Position')
            ax2.set_ylabel('Position (rad)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Joint velocity comparison (from observations)
            isaac_joint_vel = isaac_data['observations'][:min_len, 15 + i]  # joint_vel starts at index 15
            mujoco_joint_vel = mujoco_data['observations'][:min_len, 15 + i]
            
            ax3.plot(timesteps, isaac_joint_vel, label='Isaac Sim', alpha=0.7, linewidth=1)
            ax3.plot(timesteps, mujoco_joint_vel, label='mujoco', alpha=0.7, linewidth=1)
            ax3.set_title('Joint Velocity')
            ax3.set_xlabel('Timestep')
            ax3.set_ylabel('Velocity (rad/s)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{joint_name}_detailed_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        print("Detailed joint comparison plots saved")

    def generate_summary_report(self, isaac_data, mujoco_data, save_dir):
        """Generate a text summary report"""
        if isaac_data is None or mujoco_data is None:
            return
        
        min_len = min(len(isaac_data['actions']), len(mujoco_data['actions']))
        
        report_path = os.path.join(save_dir, 'comparison_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("Data Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Isaac Sim data: {len(isaac_data['actions'])} steps\n")
            f.write(f"mujoco data: {len(mujoco_data['actions'])} steps\n")
            f.write(f"Comparison length: {min_len} steps\n\n")
            
            # Action statistics
            isaac_actions = isaac_data['actions'][:min_len]
            mujoco_actions = mujoco_data['actions'][:min_len]
            
            f.write("Action Statistics:\n")
            f.write("-" * 20 + "\n")
            for i, joint_name in enumerate(self.joint_names):
                isaac_mean = np.mean(isaac_actions[:, i])
                mujoco_mean = np.mean(mujoco_actions[:, i])
                isaac_std = np.std(isaac_actions[:, i])
                mujoco_std = np.std(mujoco_actions[:, i])
                
                f.write(f"{joint_name}:\n")
                f.write(f"  Isaac - Mean: {isaac_mean:.4f}, Std: {isaac_std:.4f}\n")
                f.write(f"  mujoco  - Mean: {mujoco_mean:.4f}, Std: {mujoco_std:.4f}\n")
                f.write(f"  Diff  - Mean: {abs(isaac_mean - mujoco_mean):.4f}, Std: {abs(isaac_std - mujoco_std):.4f}\n\n")
                
            # Observation statistics for key components
            f.write("\nObservation Statistics:\n")
            f.write("-" * 25 + "\n")
            
            # Projected gravity
            isaac_gravity = isaac_data['observations'][:min_len, 0:3]
            mujoco_gravity = mujoco_data['observations'][:min_len, 0:3]
            f.write("Projected Gravity:\n")
            for i, axis in enumerate(['x', 'y', 'z']):
                isaac_mean = np.mean(isaac_gravity[:, i])
                mujoco_mean = np.mean(mujoco_gravity[:, i])
                f.write(f"  {axis} - Isaac: {isaac_mean:.4f}, mujoco: {mujoco_mean:.4f}, Diff: {abs(isaac_mean - mujoco_mean):.4f}\n")
            f.write("\n")
            
            # Velocity commands
            isaac_vel_cmd = isaac_data['observations'][:min_len, 39:42]
            mujoco_vel_cmd = mujoco_data['observations'][:min_len, 39:42]
            f.write("Velocity Commands:\n")
            for i, cmd in enumerate(['x', 'y', 'z']):
                isaac_mean = np.mean(isaac_vel_cmd[:, i])
                mujoco_mean = np.mean(mujoco_vel_cmd[:, i])
                f.write(f"  {cmd} - Isaac: {isaac_mean:.4f}, mujoco: {mujoco_mean:.4f}, Diff: {abs(isaac_mean - mujoco_mean):.4f}\n")
        
        print("Summary report saved")

def main():
    parser = argparse.ArgumentParser(description='Visualize recorded robot data')
    parser.add_argument('--isaac_data', type=str, required=True, 
                        help='Path to Isaac Sim recorded data (.pkl file)')
    parser.add_argument('--mujoco_data', type=str, required=True,
                        help='Path to mujoco logged data directory')
    parser.add_argument('--output_dir', type=str, default='visualization_output',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = DataVisualizer()
    
    # Load data
    print("Loading Isaac Sim data...")
    isaac_data = visualizer.load_isaac_data(args.isaac_data)
    
    print("Loading mujoco data...")
    mujoco_data = visualizer.load_mujoco_robot_data(args.mujoco_data)
    
    if isaac_data is None:
        print("Failed to load Isaac Sim data")
        return
    
    if mujoco_data is None:
        print("Failed to load mujoco data")
        return
    
    print("Generating visualizations...")
    
    # Generate plots
    visualizer.plot_actions(isaac_data, mujoco_data, args.output_dir)
    visualizer.plot_observations(isaac_data, mujoco_data, args.output_dir)
    visualizer.plot_statistical_comparison(isaac_data, mujoco_data, args.output_dir)
    visualizer.plot_detailed_joint_comparison(isaac_data, mujoco_data, args.output_dir)
    visualizer.generate_summary_report(isaac_data, mujoco_data, args.output_dir)
    visualizer.plot_processed_actions(isaac_data, mujoco_data, args.output_dir)
    
    print(f"All visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

# python scripts/debug/compare_records.py --isaac_data recorded_data_2025-07-15.pkl --mujoco_data scripts/debug/go1_logs_20250715_034941 --output_dir visualization_results_0715_2