LIBERO_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "gripper"]},
    },
    "observation.states.ee_state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw"]},
    },
    "observation.states.joint_state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
    },
    "observation.states.gripper_state": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["gripper", "gripper"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
    },
}

MIMICGEN_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (84,84,3),
        "names": ["height", "width", "rgb"],
    },

    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (84,84,3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "ax", "ay", "az", "gripper"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "ax", "ay", "az", "gripper"]},
    },
}

LIBERO_ABS_STANDARD_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "ax", "ay", "az", "gripper", "gripper"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "ax", "ay", "az", "gripper"]},
    },
}

MIMICGEN_ABS_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "ax", "ay", "az", "gripper_l", "gripper_r"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": ["x", "y", "z", "ax", "ay", "az", "gripper"]},
    },
}

LIBERO_ABS_QUAT_FEATURES = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]},
    },
}
