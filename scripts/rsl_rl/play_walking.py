"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import csv

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
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
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import math

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from walking_rl import RslRlOnPolicyRunnerCfg, WalkingRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../source/walking_task")))
import walking_task.tasks # noqa: F401

def stance():
    return [0, 0.65, -1.0]

def swing(t):
    return [0,-0.7 + 2.0 * math.sin(t * math.pi / 36),2.0 + 2.0 * math.sin((t+36) * math.pi / 36)]

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = WalkingRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # # export_policy_as_onnx(
    #     ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    init = 0
    t = 0
    m = 1
    # simulate environment
    csv_file = 'record_actions_test.csv'

    # with open(csv_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow([f'val_{i}' for i in range(12)])

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            hard_action_0 = stance() + stance() + stance() + stance()
            # hard_action_1 = stance() + swing(0.02*t, 1.0) + stance() + swing(0.02*t, 1.0)
            # hard_action_2 = swing(0.02*t, 1.0) + stance() + swing(0.02*t, 1.0) + stance()
            # if m == 1:
            #     hard_action_tensor_1 = torch.tensor(hard_action_1, dtype=torch.float32)
            #     hard_action_tensor = hard_action_tensor_1.repeat(256, 1).to('cuda:0')
            # else:
            #     hard_action_tensor_1 = torch.tensor(hard_action_2, dtype=torch.float32)
            #     hard_action_tensor = hard_action_tensor_1.repeat(256, 1).to('cuda:0')

            # hard_action_tensor_1 = torch.tensor(hard_action_0, dtype=torch.float32)
            # hard_action_tensor = hard_action_tensor_1.repeat(256, 1).to('cuda:0')
            
            # print(f"[INFO] Hard Actions: {hard_action_tensor}")
            # print(f"[INFO] Hard Actions shape: {len(hard_action_tensor)}")

            # with open(csv_file, mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(actions.squeeze().tolist())

            t += 1
            # if t > 50:
            #     t = 0
            #     m = -m
            # env stepping
            if init <= 50:
                obs, _, _, _ = env.step(actions)
                init += 1
                # last_action = actions
            else:
                # obs, _, _, _ = env.step(last_action)
                hard_action_test_1 = swing(t % 36)
                hard_action_test_2 = swing((t+18) % 36)

                actions[:, 0] = 0.0 #hard_action_test[0]
                actions[:, 1] = hard_action_test_1[1]
                actions[:, 2] = hard_action_test_1[2]
                actions[:, 3] = 0.0
                actions[:, 4] = hard_action_test_2[1]
                actions[:, 5] = hard_action_test_2[2]
                actions[:, 6] = 0.0
                actions[:, 7] = hard_action_test_2[1]
                actions[:, 8] = hard_action_test_2[2]
                actions[:, 9] = 0.0
                actions[:, 10] = hard_action_test_1[1]
                actions[:, 11] = hard_action_test_1[2]
                # with open(csv_file, mode='a', newline='') as file:
                #     print(f"[INFO] Writing actions to {csv_file}")
                #     writer = csv.writer(file)
                #     writer.writerow(actions.squeeze().tolist())
                # print(f"[INFO] Actions: {actions}")
                # print(f"[INFO] Actions shape: {actions.shape}")
                obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
