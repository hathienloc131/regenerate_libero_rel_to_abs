#!/usr/bin/env python3
"""
Re-render MimicGen HDF5 datasets at a new image resolution.

For each demo, the exact recorded MuJoCo states are restored step-by-step
(state-forcing), images are captured at the target resolution, and the
result is written to a new HDF5 file with identical structure except for
the image observations.

Usage:
    python replay_mimicgen_dataset.py --task coffee
    python replay_mimicgen_dataset.py --task coffee --max-demos 10 --render
    python replay_mimicgen_dataset.py --workers 4   # all tasks in parallel
    python replay_mimicgen_dataset.py               # all tasks sequentially
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Optional

# mimicgen_envs lives inside the local mimicgen/ repo checkout.
sys.path.insert(0, str(Path(__file__).parent / "mimicgen"))
# robosuite package lives inside the local robosuite/ repo checkout.
sys.path.insert(0, str(Path(__file__).parent / "robosuite"))

import h5py
import numpy as np
import robosuite


# ── Task mappings ─────────────────────────────────────────────────────────────

TASK_TO_ENV: dict[str, str] = {
    # "coffee":               "Coffee_D2",                #running 
    # "coffee_preparation":   "CoffeePreparation_D1",     #running
    # "hammer_cleanup":       "HammerCleanup_D1",         #running
    # "kitchen":              "Kitchen_D1",               #done
    # "mug_cleanup":          "MugCleanup_D1",            #running
    "nut_assembly":         "NutAssembly_D0",           #done
    "pick_place":           "PickPlace_D0",             #done
    # "square":               "Square_D2",                #done
    # "stack":                "Stack_D1",                 #running
    # "stack_three":          "StackThree_D1",
    # "threading":            "Threading_D2",
    # "three_piece_assembly": "ThreePieceAssembly_D2",
}

TASK_TO_HDF5: dict[str, str] = {
    # "coffee":               "coffee_d2.hdf5",
    # "coffee_preparation":   "coffee_preparation_d1.hdf5",
    # "hammer_cleanup":       "hammer_cleanup_d1.hdf5",
    # "kitchen":              "kitchen_d1.hdf5",
    # "mug_cleanup":          "mug_cleanup_d1.hdf5",
    "nut_assembly":         "nut_assembly_d0.hdf5",
    "pick_place":           "pick_place_d0.hdf5",
    # "square":               "square_d2.hdf5",
    # "stack":                "stack_d1.hdf5",
    # "stack_three":          "stack_three_d1.hdf5",
    # "threading":            "threading_d2.hdf5",
    # "three_piece_assembly": "three_piece_assembly_d2.hdf5",
}

IMAGE_OBS    = ["agentview_image", "robot0_eye_in_hand_image"]
CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]


# ── Environment ───────────────────────────────────────────────────────────────

def make_env(env_name: str, img_size: int, render: bool, robots="Panda"):
    from robosuite.controllers import load_controller_config
    import mimicgen_envs  # noqa: F401 — registers MimicGen envs

    ctrl = load_controller_config(default_controller="OSC_POSE")
    return robosuite.make(
        env_name=env_name,
        robots=robots,
        controller_configs=ctrl,
        has_renderer=render,
        has_offscreen_renderer=True,
        use_camera_obs=False,   # we render directly via sim.render — no observable overhead
        camera_names=CAMERA_NAMES,
        camera_heights=img_size,
        camera_widths=img_size,
        control_freq=20,
        reward_shaping=False,
        ignore_done=False,
    )


def _collect_expected_geoms(env) -> set:
    """Return all geom names that env.model.generate_id_mappings will look up."""
    def flatten(lst):
        out = []
        for item in lst:
            if isinstance(item, list):
                out.extend(flatten(item))
            elif item is not None:
                out.append(item)
        return out

    models = list(env.model.mujoco_objects)
    for robot in env.model.mujoco_robots:
        models.append(robot)
        models.extend(robot.models)

    geoms = set()
    for m in models:
        geoms.update(flatten(getattr(m, 'visual_geoms', [])))
        geoms.update(flatten(getattr(m, 'contact_geoms', [])))
    return geoms


def restore_state(env, model_xml: str, mujoco_state: np.ndarray):
    """Load scene XML then set the exact MuJoCo qpos/qvel."""
    import re
    env.reset()

    # Collect all geom names the model expects before we swap out the XML.
    expected_geoms = _collect_expected_geoms(env)

    version_minor = int(robosuite.__version__.split(".")[1])
    xml = (env.edit_model_xml(model_xml) if version_minor >= 4
           else _postprocess_xml(model_xml))

    for bad_body in ("robot0_head_1",):
        xml = re.sub(
            rf'(<body\b[^>]*\bname="{bad_body}"[^>]*>)',
            r'\1<inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>',
            xml,
        )

    # Old dataset XMLs (recorded with an earlier robosuite) may be missing geoms
    # that the current model expects in generate_id_mappings. Inject any that are
    # absent as tiny invisible spheres so the id-lookup doesn't raise.
    present_geoms = set(re.findall(r'<geom\b[^>]*\bname="([^"]+)"', xml))
    missing = expected_geoms - present_geoms
    if missing:
        dummy = "".join(
            f'<geom name="{name}" type="sphere" size="0.001" pos="0 0 -100"'
            f' contype="0" conaffinity="0" group="0"/>'
            for name in missing
        )
        xml = xml.replace("</worldbody>", dummy + "</worldbody>", 1)

    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(mujoco_state)
    env.sim.forward()


def _postprocess_xml(xml: str) -> str:
    from robosuite.utils.mjcf_utils import postprocess_model_xml
    return postprocess_model_xml(xml)


# ── Per-demo re-render ────────────────────────────────────────────────────────

def rerender_demo(
    env,
    src_grp: h5py.Group,
    dst_grp: h5py.Group,
    img_size: int,
    render: bool,
) -> bool:
    """
    State-force every timestep, render images directly via sim.render(),
    and write to dst_grp.

    Using sim.render() directly is significantly faster than
    _get_observations(force_update=True) because it skips updating all
    non-camera observables (joint sensors, eef computations, etc.).
    """
    model_xml = src_grp.attrs["model_file"]
    states    = src_grp["states"][()]   # (T, state_dim)
    T         = len(states)

    # copy attrs and non-obs datasets verbatim
    for k, v in src_grp.attrs.items():
        dst_grp.attrs[k] = v
    for key in src_grp.keys():
        if key != "obs":
            src_grp.copy(key, dst_grp)

    # determine non-image obs keys from source — re-extract these from sim too
    src_obs_keys = list(src_grp["obs"].keys()) if "obs" in src_grp else []
    non_img_keys = [k for k in src_obs_keys if k not in IMAGE_OBS]

    # pre-allocate all buffers up front
    img_buffers = {
        obs_key: np.empty((T, img_size, img_size, 3), dtype=np.uint8)
        for obs_key in IMAGE_OBS
    }
    non_img_buffers = {
        key: np.empty(src_grp["obs"][key].shape, dtype=src_grp["obs"][key].dtype)
        for key in non_img_keys
    }

    restore_state(env, model_xml, states[0])

    for t in range(T):
        env.sim.set_state_from_flattened(states[t])
        env.sim.forward()

        # render images directly — skips camera observable pipeline
        for cam_name, obs_key in zip(CAMERA_NAMES, IMAGE_OBS):
            img_buffers[obs_key][t] = np.array(
                env.sim.render(camera_name=cam_name, width=img_size, height=img_size)[::-1]
            )

        # re-extract non-image obs from sim state for consistency
        # use_camera_obs=False so this updates joints/eef/task obs without rendering
        if non_img_keys:
            obs = env._get_observations(force_update=True)
            for key in non_img_keys:
                if key in obs:
                    non_img_buffers[key][t] = obs[key]

        if render:
            env.render()

    obs_grp = dst_grp.require_group("obs")

    # write re-rendered images (lzf is faster than gzip; fall back if unavailable)
    for obs_key, arr in img_buffers.items():
        try:
            obs_grp.create_dataset(obs_key, data=arr, compression="lzf", shuffle=True)
        except ValueError:
            obs_grp.create_dataset(obs_key, data=arr, compression="gzip", compression_opts=1)

    # write re-extracted non-image obs
    for key, arr in non_img_buffers.items():
        obs_grp.create_dataset(key, data=arr)

    rewards = src_grp["rewards"][()] if "rewards" in src_grp else np.zeros(T)
    success = bool(rewards[-1] > 0 or
                   (src_grp["dones"][()] if "dones" in src_grp else np.zeros(T))[-1])
    return success


# ── Per-task pipeline ─────────────────────────────────────────────────────────

def process_task(
    task: str,
    src_dir: Path,
    dst_dir: Path,
    img_size: int,
    max_demos: Optional[int],
    render: bool,
):
    env_name  = TASK_TO_ENV.get(task)
    hdf5_name = TASK_TO_HDF5.get(task)

    if env_name is None or hdf5_name is None:
        print(f"  [skip] unknown task '{task}'")
        return

    src_path = src_dir / hdf5_name
    dst_path = dst_dir / hdf5_name

    if not src_path.exists():
        print(f"  [skip] source not found: {src_path}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src_f, h5py.File(dst_path, "w") as dst_f:
        robots = json.loads(src_f["data"].attrs["env_args"])["env_kwargs"].get("robots", ["Panda"])
        env = make_env(env_name, img_size=img_size, render=render, robots=robots)

        demos = sorted(src_f["data"].keys(), key=lambda k: int(k[5:]))
        if max_demos is not None:
            demos = demos[:max_demos]

        print(f"  {task}: {len(demos)} demos  →  {dst_path}", flush=True)

        n_success = 0
        for demo_key in demos:
            src_grp = src_f[f"data/{demo_key}"]
            dst_grp = dst_f.require_group(f"data/{demo_key}")

            success = rerender_demo(env, src_grp, dst_grp, img_size=img_size, render=render)
            n_success += success
            status = "✓" if success else "✗"
            T = src_grp["states"].shape[0]
            print(f"    [{task}] {demo_key}: {status}  T={T}", flush=True)

        # copy data-group attrs; update camera resolution inside env_args JSON
        for k, v in src_f["data"].attrs.items():
            if k == "env_args":
                env_args = json.loads(v)
                env_args["env_kwargs"]["camera_heights"] = img_size
                env_args["env_kwargs"]["camera_widths"]  = img_size
                dst_f["data"].attrs[k] = json.dumps(env_args, indent=4)
            elif k == "total":
                total = sum(
                    int(dst_f[f"data/{d}"].attrs.get("num_samples", 0))
                    for d in dst_f["data"].keys()
                )
                dst_f["data"].attrs["total"] = total
            else:
                dst_f["data"].attrs[k] = v

        # copy top-level masks/metadata if present
        for key in src_f.keys():
            if key != "data":
                src_f.copy(key, dst_f)

    env.close()
    print(f"  → {task}: {n_success}/{len(demos)} success  ({100*n_success/max(len(demos),1):.1f}%)",
          flush=True)


# ── Multiprocessing worker (must be top-level for pickling) ──────────────────

def _worker(args: tuple):
    task, src_dir, dst_dir, img_size, max_demos = args
    process_task(task, Path(src_dir), Path(dst_dir), img_size, max_demos, render=False)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-render MimicGen HDF5 datasets at a new image resolution"
    )
    parser.add_argument("--src-dir",   default="data/mimicgen",     help="Source HDF5 directory")
    parser.add_argument("--dst-dir",   default="data/mimicgen_256", help="Output HDF5 directory")
    parser.add_argument("--img-size",  type=int, default=256,       help="Output image resolution (square)")
    parser.add_argument("--task",      default=None,                help="Single task key, e.g. 'coffee'")
    parser.add_argument("--max-demos", type=int, default=None,      help="Max demos per task")
    parser.add_argument("--render",    action="store_true",         help="Show live viewer (forces --workers 1)")
    parser.add_argument("--workers",   type=int, default=1,
                        help="Number of tasks to process in parallel (default: 1)")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    tasks   = [args.task] if args.task else list(TASK_TO_HDF5.keys())

    if args.render:
        args.workers = 1  # viewer can't run in a subprocess

    print(f"Source:     {src_dir}")
    print(f"Output:     {dst_dir}")
    print(f"Image size: {args.img_size}×{args.img_size}")
    print(f"Tasks:      {tasks}")
    print(f"Workers:    {args.workers}\n")

    if args.workers > 1:
        worker_args = [
            (task, str(src_dir), str(dst_dir), args.img_size, args.max_demos)
            for task in tasks
        ]
        # spawn is required on macOS to avoid MuJoCo/OpenGL fork issues
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            pool.map(_worker, worker_args)
    else:
        for task in tasks:
            print(f"{'='*60}\nTask: {task}")
            process_task(
                task=task,
                src_dir=src_dir,
                dst_dir=dst_dir,
                img_size=args.img_size,
                max_demos=args.max_demos,
                render=args.render,
            )


if __name__ == "__main__":
    main()
