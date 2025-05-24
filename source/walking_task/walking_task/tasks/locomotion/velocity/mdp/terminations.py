# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return False  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")
    
def roll_exceed(env: ManagerBasedRLEnv, roll_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the roll of the robot exceeds a certain threshold.

    This is used to prevent the robot from rolling over.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the roll
    sinr_cosp = 2 * (asset.data.root_quat_w[:,0] * asset.data.root_quat_w[:,1] + asset.data.root_quat_w[:,2] * asset.data.root_quat_w[:,3])
    cosr_cosp = 1 - 2 * (asset.data.root_quat_w[:,1] * asset.data.root_quat_w[:,1] + asset.data.root_quat_w[:, 2] * asset.data.root_quat_w[:,2])
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    # print(f"roll: {roll}")

    return torch.abs(roll) > roll_threshold

def pitch_exceed(env: ManagerBasedRLEnv, pitch_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the pitch of the robot exceeds a certain threshold.

    This is used to prevent the robot from pitching over.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the pitch
    sinp = 2 * (asset.data.root_quat_w[:,0] * asset.data.root_quat_w[:,2] - asset.data.root_quat_w[:,3] * asset.data.root_quat_w[:,1])
    cosp = torch.sqrt(1 - sinp * sinp)
    pitch = torch.atan2(sinp, cosp)
    return torch.abs(pitch) > pitch_threshold