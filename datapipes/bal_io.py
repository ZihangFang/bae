import os

import numpy as np
import torch

DTYPE = torch.float64


def _rotvec_to_quat_xyzw(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half_theta = 0.5 * theta
    cos_half = torch.cos(half_theta)
    sin_half = torch.sin(half_theta)

    eps = 1e-12
    scale = torch.where(theta > eps, sin_half / theta, 0.5 - (theta * theta) / 48.0)
    xyz = rotvec * scale
    return torch.cat([xyz, cos_half], dim=-1)


def read_bal_data(file_name: str, use_quat: bool = True) -> dict:
    """
    Read a Bundle Adjustment in the Large dataset problem text file.

    Format:
      <num_cameras> <num_points> <num_observations>
      <camera_index> <point_index> <x> <y>    (repeated n_observations)
      <camera parameters>                    (n_cameras * 9 lines)
      <point parameters>                     (n_points * 3 lines)

    Each camera has 9 parameters: Rodrigues rotvec (3), translation (3), f, k1, k2.
    This loader outputs either:
      - use_quat=True:  [tx, ty, tz, qx, qy, qz, qw, f, k1, k2] (10)
      - use_quat=False: [tx, ty, tz, rx, ry, rz, f, k1, k2]  (9)
    """
    with open(file_name, "r") as file:
        n_cameras, n_points, n_observations = map(int, file.readline().split())
        values = np.fromfile(file, sep=" ", dtype=np.float64)

    observation_value_count = n_observations * 4
    camera_value_count = n_cameras * 9
    point_value_count = n_points * 3
    expected_value_count = observation_value_count + camera_value_count + point_value_count
    if values.size != expected_value_count:
        raise ValueError(
            f"Expected {expected_value_count} numeric values in BAL file, parsed {values.size}."
        )

    observations = values[:observation_value_count].reshape(n_observations, 4)
    camera_indices = torch.from_numpy(observations[:, 0].astype(np.int64, copy=True))
    point_indices = torch.from_numpy(observations[:, 1].astype(np.int64, copy=True))
    points_2d = torch.from_numpy(observations[:, 2:4].copy()).to(DTYPE)

    camera_start = observation_value_count
    point_start = camera_start + camera_value_count
    camera_params = torch.from_numpy(
        values[camera_start:point_start].reshape(n_cameras, 9).copy()
    ).to(DTYPE)
    points_3d = torch.from_numpy(
        values[point_start:].reshape(n_points, 3).copy()
    ).to(DTYPE)

    if use_quat:
        q = _rotvec_to_quat_xyzw(camera_params[:, :3])
        camera_params = torch.cat([camera_params[:, 3:6], q, camera_params[:, 6:]], dim=1)
    else:
        camera_params = torch.cat([camera_params[:, 3:6], camera_params[:, :3], camera_params[:, 6:]], dim=1)

    return {
        "problem_name": os.path.splitext(os.path.basename(file_name))[0],
        "camera_params": camera_params.to(DTYPE),
        "points_3d": points_3d,
        "points_2d": points_2d,
        "camera_index_of_observations": camera_indices,
        "point_index_of_observations": point_indices,
    }
