"""Script to play a checkpoint if an RL agent from RSL-RL and record actions and obs."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np
import pickle

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

# Import extensions to set up environment tasks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../source/walking_task")))
import walking_task.tasks # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # specify directory for logging experiments
    log_path = os.path.join("logs", "rsl_rl", "unitree_go1_walk_0704", "2025-07-06_09-29-38", "exported", "policy.pt")

    # Check if model file exists
    if not os.path.exists(log_path):
        print(f"[ERROR]: Model checkpoint not found at: {log_path}")
        return

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode= None)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {log_path}")
    try:
        # load previously trained model
        model = torch.jit.load(log_path)
        model.eval()
        device = torch.device(args_cli.device if args_cli.device else "cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    except Exception as e:
        print(f"[ERROR]: Failed to load model: {e}")
        env.close()
        return

    # reset environment
    # env.reset()
    obs, _ = env.get_observations()
    obs = obs.to(device)
    print(f"[INFO]: Observation: {obs}")

    # Initialize data recording
    max_steps = 1000
    recorded_data = {
        'actions': [],
        'observations': [],
        'timesteps': [],
        'metadata': {
            'total_steps': max_steps,
            'task': args_cli.task,
            'model_path': log_path,
            'obs_dim': obs.shape[-1] if obs is not None else None,
        }
    }
    # simulate environment
    for timestep in range(max_steps):
        if not simulation_app.is_running():
            print(f"[INFO]: Simulation stopped at step {timestep}")
            break
            
        # run everything in inference mode
        with torch.no_grad():
            # agent stepping
            obs = obs.to(device)
            actions = model(obs)
            # print(f"[INFO]: Actions: {actions}")
            
            # Record data (convert to numpy for easier analysis)
            recorded_data['actions'].append(actions.cpu().numpy().copy())
            recorded_data['observations'].append(obs.cpu().numpy().copy())
            recorded_data['timesteps'].append(timestep)
            
            # env stepping
            obs, _, _, _ = env.step(actions)
            
            # # Check if episode is done
            # if terminated or truncated:
            #     print(f"[INFO]: Episode ended at step {timestep}, resetting...")
            #     obs, _ = env.reset()
            #     obs = obs.to(device)
        
        # Progress indicator
        if (timestep + 1) % 100 == 0:
            print(f"[INFO]: Completed {timestep + 1}/{max_steps} steps")

    # Convert lists to numpy arrays for easier analysis
    recorded_data['actions'] = np.array(recorded_data['actions'])
    recorded_data['observations'] = np.array(recorded_data['observations'])
    recorded_data['timesteps'] = np.array(recorded_data['timesteps'])
    
    # Update metadata with final dimensions
    recorded_data['metadata']['action_shape'] = recorded_data['actions'].shape
    recorded_data['metadata']['obs_shape'] = recorded_data['observations'].shape
    recorded_data['metadata']['actual_steps'] = len(recorded_data['timesteps'])
    
    # # Save data
    # save_path = f"recorded_data_2025-07-06_09-29-38.pkl"
    
    # with open(save_path, 'wb') as f:
    #     pickle.dump(recorded_data, f)
    
    print(f"[INFO]: Data saved to {save_path}")
    print(f"[INFO]: Recorded {len(recorded_data['timesteps'])} steps")
    print(f"[INFO]: Action shape: {recorded_data['actions'].shape}")
    print(f"[INFO]: Observation shape: {recorded_data['observations'].shape}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()