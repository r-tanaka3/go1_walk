import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def load_recorded_data(file_path: str) -> Dict:
    """Load recorded data from pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def define_observation_categories() -> Dict[str, Tuple[int, int]]:
    """
    Define observation subcategories with their index ranges.
    Modify these ranges based on your specific task's observation structure.
    """
    categories = {
        'projected_gravity': (0, 3),
        'joint_pos': (3, 15),
        'joint_vel': (15, 27),
        'actions': (27, 39),
        'velocity_commands': (39, 42),
    }
    return categories

def extract_observation_subcategory(observations: np.ndarray, category: str, 
                                  categories: Dict[str, Tuple[int, int]]) -> np.ndarray:
    """Extract specific subcategory from observations."""
    start_idx, end_idx = categories[category]
    return observations[:, :, start_idx:end_idx]  # shape: (steps, envs, features)

def plot_actions(actions: np.ndarray, save_path: str = None):
    """Plot action time series."""
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()
    
    # Assuming 12 actions for quadruped (3 per leg)
    action_names = [
        'FL_hip', 'FL_thigh', 'FL_calf',
        'FR_hip', 'FR_thigh', 'FR_calf', 
        'RL_hip', 'RL_thigh', 'RL_calf',
        'RR_hip', 'RR_thigh', 'RR_calf'
    ]
    
    for i in range(min(12, actions.shape[-1])):
        if i < len(axes):
            # Plot for first environment (env 0)
            axes[i].plot(actions[:, 0, i])
            axes[i].set_title(f'Action {i}: {action_names[i] if i < len(action_names) else f"Action_{i}"}')
            axes[i].set_xlabel('Timestep')
            axes[i].set_ylabel('Action Value')
            axes[i].grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_observation_category(observations: np.ndarray, category: str, 
                            categories: Dict[str, Tuple[int, int]], save_path: str = None):
    """Plot specific observation category."""
    cat_data = extract_observation_subcategory(observations, category, categories)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(min(4, cat_data.shape[-1])):
        if i < len(axes):
            # Plot for first environment (env 0)
            axes[i].plot(cat_data[:, 0, i])
            axes[i].set_title(f'{category} - Dim {i}')
            axes[i].set_xlabel('Timestep')
            axes[i].set_ylabel('Value')
            axes[i].grid(True)
    
    plt.suptitle(f'{category} Time Series')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load data
    data = load_recorded_data("recorded_data_2025-07-06_09-29-38.pkl")
    
    # Define observation categories
    obs_categories = define_observation_categories()
    
    # Plot actions
    plot_actions(data['actions'], 'actions_plot.png')
    
    # Plot different observation categories
    for category in obs_categories.keys():
        plot_observation_category(data['observations'], category, obs_categories, 
                                    f'{category}_plot.png')
    
    print("Data analysis completed!")