from pathlib import Path

import numpy as np
from h5py import File
from scipy.spatial.transform import Rotation
import math


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def load_local_episodes(input_h5: Path, max_episodes: int = None):
    """Load episodes from regenerate_libero_abs format HDF5 with axis-angle representation.

    Reads obs keys: agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori (axis-angle), gripper_states.
    Actions are absolute [abs_x, abs_y, abs_z, ax, ay, az, gripper] (axis-angle orientation).
    Output state: [abs_x, abs_y, abs_z, ax, ay, az, gripper] (7-dim).
    Output action: [abs_x, abs_y, abs_z, ax, ay, az, gripper] (7-dim).
    """
    with File(input_h5, "r") as f:
        demos = list(f["data"].values())
        if max_episodes is not None:
            demos = demos[:max_episodes]
        for demo in demos:
            demo_len = len(demo["obs/agentview_rgb"])
            # Actions: absolute [abs_x, abs_y, abs_z, ax, ay, az, gripper]
            raw_action = np.array(demo["actions"])
            # gripper: (-1: open, 1: close) -> (0: close, 1: open)
            gripper_action = (1 - np.clip(raw_action[:, -1], 0, 1))[:, None]
            # Normalize action axis-angle via quat (w >= 0) to remove sign-flip discontinuities
            action_quat = Rotation.from_rotvec(raw_action[:, 3:6]).as_quat()  # [x, y, z, w]
            action_quat[action_quat[:, 3] < 0] *= -1  # enforce w >= 0
            action_axisangle = Rotation.from_quat(action_quat).as_rotvec()
            action = np.concatenate(
                [raw_action[:, :3], action_axisangle, gripper_action],
                axis=1,
                dtype=np.float32,
            )
            # State: ee_pos (3) + ee_ori axis-angle (3) + gripper mean (1) = 7 dims
            ee_pos = np.array(demo["obs/ee_pos"])
            ee_ori = np.array(demo["obs/ee_ori"])
            gripper_avg = np.mean(demo["obs/gripper_states"], axis=1, keepdims=True)
            state = np.concatenate([ee_pos, ee_ori, gripper_avg], axis=1)

            imgs = np.array(demo["obs/agentview_rgb"])[:, ::-1, ::-1, :]
            wrist_imgs = np.array(demo["obs/eye_in_hand_rgb"])[:, ::-1, ::-1, :]
            episode = {
                "observation.images.image": imgs,
                "observation.images.wrist_image": wrist_imgs,
                "observation.state": np.array(state, dtype=np.float32),
                "action": np.array(action, dtype=np.float32),
            }
            yield [{**{k: v[i] for k, v in episode.items()}} for i in range(demo_len)]


def load_local_episodes_abs_quat(input_h5: Path, max_episodes: int = None):
    """Load episodes from regenerate_libero_abs format HDF5 with quaternion representation.

    Reads obs keys: agentview_rgb, eye_in_hand_rgb, ee_pos, ee_ori (axis-angle), gripper_states.
    Actions are absolute [abs_x, abs_y, abs_z, ax, ay, az, gripper] (axis-angle orientation).
    Converts axis-angle orientation to quaternion [qx, qy, qz, qw] for both state and action.
    Output action: [abs_x, abs_y, abs_z, qx, qy, qz, qw, gripper] (8-dim).
    """
    with File(input_h5, "r") as f:
        demos = list(f["data"].values())
        if max_episodes is not None:
            demos = demos[:max_episodes]
        for demo in demos:
            demo_len = len(demo["obs/agentview_rgb"])

            # Actions: absolute [abs_x, abs_y, abs_z, ax, ay, az, gripper]
            raw_action = np.array(demo["actions"])
            # gripper: (-1: open, 1: close) -> (0: close, 1: open)
            gripper_action = (1 - np.clip(raw_action[:, -1], 0, 1))[:, None]
            # Convert axis-angle orientation to quaternion [qx, qy, qz, qw], enforce w >= 0
            action_quat = Rotation.from_rotvec(raw_action[:, 3:6]).as_quat()  # [x, y, z, w]
            action_quat[action_quat[:, 3] < 0] *= -1
            action = np.concatenate(
                [raw_action[:, :3], action_quat, gripper_action],
                axis=1,
                dtype=np.float32,
            )

            # State: ee_pos (3) + ee_ori axis-angle->quat (4) + gripper mean (1) = 8 dims, enforce w >= 0
            ee_pos = np.array(demo["obs/ee_pos"])
            ee_quat = Rotation.from_rotvec(np.array(demo["obs/ee_ori"])).as_quat()  # [x, y, z, w]
            ee_quat[ee_quat[:, 3] < 0] *= -1
            gripper_avg = np.mean(demo["obs/gripper_states"], axis=1, keepdims=True)
            state = np.concatenate([ee_pos, ee_quat, gripper_avg], axis=1)

            imgs = np.array(demo["obs/agentview_rgb"])[:, ::-1, ::-1, :]
            wrist_imgs = np.array(demo["obs/eye_in_hand_rgb"])[:, ::-1, ::-1, :]
            episode = {
                "observation.images.image": imgs,
                "observation.images.wrist_image": wrist_imgs,
                "observation.state": np.array(state, dtype=np.float32),
                "action": np.array(action, dtype=np.float32),
            }
            yield [{**{k: v[i] for k, v in episode.items()}} for i in range(demo_len)]
