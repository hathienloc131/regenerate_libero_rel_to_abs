from pathlib import Path

import cv2
import numpy as np
from h5py import File
from scipy.spatial.transform import Rotation
import math


def _decode_jpeg_array(raw_array):
    """Decode an array of JPEG-encoded bytes into an (N, H, W, 3) RGB uint8 array."""
    frames = []
    for buf in raw_array:
        img = cv2.imdecode(np.frombuffer(bytes(buf), dtype=np.uint8), cv2.IMREAD_COLOR)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def _load_images(demo, key_raw, key_jpeg):
    """Return decoded image array, trying raw key first then JPEG fallback."""
    if key_raw in demo["obs"]:
        return np.array(demo[f"obs/{key_raw}"])
    return _decode_jpeg_array(demo[f"obs/{key_jpeg}"])


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
            imgs = _load_images(demo, "agentview_rgb", "agentview_rgb_jpeg")
            wrist_imgs = _load_images(demo, "eye_in_hand_rgb", "eye_in_hand_rgb_jpeg")
            demo_len = len(imgs)
            # (-1: open, 1: close) -> (0: close, 1: open)
            action = np.array(demo["actions"])
            action = np.concatenate(
                [
                    action[:, :6],
                    (1 - np.clip(action[:, -1], 0, 1))[:, None],
                ],
                axis=1,
            )
            state = np.concatenate(
                [
                    np.array(demo["obs/ee_states"]),
                    np.array(demo["obs/gripper_states"]),
                ],
                axis=1,
            )
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
            imgs = _load_images(demo, "agentview_rgb", "agentview_rgb_jpeg")
            wrist_imgs = _load_images(demo, "eye_in_hand_rgb", "eye_in_hand_rgb_jpeg")
            demo_len = len(imgs)

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

            episode = {
                "observation.images.image": imgs[:, ::-1, ::-1, :],
                "observation.images.wrist_image": wrist_imgs[:, ::-1, ::-1, :],
                "observation.state": np.array(state, dtype=np.float32),
                "action": np.array(action, dtype=np.float32),
            }
            yield [{**{k: v[i] for k, v in episode.items()}} for i in range(demo_len)]

def load_local_episodes_mimicgen_abs(img_h5: Path, abs_h5: Path, max_episodes: int = None):
    """Load episodes combining 256×256 images from img_h5 and absolute actions from abs_h5.

    Images  : agentview_image, robot0_eye_in_hand_image  — from img_h5 (mimicgen_256)
    Actions : absolute [x, y, z, ax, ay, az, gripper]    — from abs_h5 (mimicgen_abs_256)
    State   : eef_pos(3) + rotvec(3) + gripper_qpos(2)   — from abs_h5 proprioception obs
    """
    with File(img_h5, "r") as img_f, File(abs_h5, "r") as abs_f:
        demo_keys = sorted(img_f["data"].keys(), key=lambda k: int(k[5:]))
        if max_episodes is not None:
            demo_keys = demo_keys[:max_episodes]
        for demo_key in demo_keys:
            img_demo = img_f[f"data/{demo_key}"]
            abs_demo = abs_f[f"data/{demo_key}"]

            demo_len = len(img_demo["obs/agentview_image"])

            # absolute actions: [x, y, z, ax, ay, az, gripper]
            raw_action = np.array(abs_demo["actions"])
            gripper_action = (1 - np.clip(raw_action[:, -1], 0, 1))[:, None]
            action = np.concatenate(
                [raw_action[:, :3], raw_action[:, 3:6], gripper_action],
                axis=1,
                dtype=np.float32,
            )

            # state: eef_pos(3) + rotvec(3) + gripper_qpos(2) = 8 dims
            ee_pos = np.array(abs_demo["obs/robot0_eef_pos"])
            ee_quat = np.array(abs_demo["obs/robot0_eef_quat"])  # [x, y, z, w]
            ee_rotvec = Rotation.from_quat(ee_quat).as_rotvec()
            gripper_qpos = np.array(abs_demo["obs/robot0_gripper_qpos"])
            state = np.concatenate([ee_pos, ee_rotvec, gripper_qpos], axis=1).astype(np.float32)

            episode = {
                "observation.images.image": np.array(img_demo["obs/agentview_image"]),
                "observation.images.wrist_image": np.array(img_demo["obs/robot0_eye_in_hand_image"]),
                "observation.state": state,
                "action": action,
            }
            yield [{k: v[i] for k, v in episode.items()} for i in range(demo_len)]


def load_local_episodes_mimicgen(input_h5: Path, max_episodes: int = None):
    """Load episodes from MimicGen HDF5 with RPY state representation.

    Actions are delta [dx, dy, dz, dax, day, daz, gripper] (OSC_POSE axis-angle).
    Output state: [eef_pos(3), rpy(3), gripper_qpos(2)] = 8 dims.
    Output action: [dx, dy, dz, dax, day, daz, gripper] = 7 dims,
                   gripper: (-1: open, 1: close) -> (0: close, 1: open).
    """
    with File(input_h5, "r") as f:
        demos = list(f["data"].values())
        if max_episodes is not None:
            demos = demos[:max_episodes]
        for demo in demos:
            demo_len = len(demo["obs/agentview_image"])

            raw_action = np.array(demo["actions"])
            gripper_action = (1 - np.clip(raw_action[:, -1], 0, 1))[:, None]
            
            axisangle_action = raw_action[:, 3:6]
            action = np.concatenate(
                [raw_action[:, :3], axisangle_action, gripper_action],
                axis=1,
                dtype=np.float32,
            )

            # State: eef_pos (3) + axisangle (3) + gripper_qpos (2) = 8 dims
            # robot0_eef_quat is [x, y, z, w] (robosuite convention)
            ee_pos = np.array(demo["obs/robot0_eef_pos"])
            ee_quat = np.array(demo["obs/robot0_eef_quat"])  # [x, y, z, w]
            ee_rpy = Rotation.from_quat(ee_quat).as_rotvec()
            gripper_qpos = np.array(demo["obs/robot0_gripper_qpos"])
            print("ee_pos:", ee_pos.shape, "ee_quat:", ee_quat.shape, "ee_rpy:", ee_rpy.shape, "gripper_qpos:", gripper_qpos.shape)
            state = np.concatenate([ee_pos, ee_rpy, gripper_qpos], axis=1)

            episode = {
                "observation.images.image": np.array(demo["obs/agentview_image"]),
                "observation.images.wrist_image": np.array(demo["obs/robot0_eye_in_hand_image"]),
                "observation.state": np.array(state, dtype=np.float32),
                "action": np.array(action, dtype=np.float32),
            }
            yield [{**{k: v[i] for k, v in episode.items()}} for i in range(demo_len)]
