from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import ContactSensor
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def feet_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """ Observation of foot contact indicators.

    Returns binarized foot contact indicators (4 dimensions)
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute norm of forces for each foot
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # shape: (N, 4, 3)
    force_norms = torch.norm(forces, dim=2)  # shape: (N, 4)
    
    # compare norm with threshold
    feet_contact_bi = force_norms > threshold
    return feet_contact_bi