# LIBERO: Relative to Absolute Action Conversion

A two-step pipeline to convert LIBERO robot manipulation datasets from **relative** end-effector actions to **absolute** actions, then package them into [LeRobot](https://github.com/huggingface/lerobot) dataset format.

## Overview

**Step 1 — `regenerate_libero_dataset.py`**
Replays original LIBERO demonstrations in simulation to produce new HDF5 files with:
- Absolute end-effector actions (position + axis-angle orientation + gripper)
- Higher resolution images (256×256 instead of 128×128)
- No-op actions filtered out
- Failed demonstrations filtered out

**Step 2 — `libero_h5.py`**
Converts the regenerated HDF5 files into LeRobot dataset format. Supports parallel processing (local or Ray) and optional upload to Hugging Face Hub.

## Requirements

Python 3.10+

```
h5py
numpy
scipy
tqdm
pandas
robosuite
libero
lerobot
ray
datatrove
```

Install dependencies:

```bash
pip install h5py numpy scipy tqdm pandas ray datatrove
```

Install LIBERO (includes robosuite):
```bash
pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

Install LeRobot:
```bash
pip install lerobot
```

## Usage

### Step 1: Regenerate dataset with absolute actions

```bash
python regenerate_libero_dataset.py \
    --libero_task_suite libero_10 \
    --libero_raw_data_dir ./data/raw/libero_10_raw \
    --libero_target_dir ./data/processed/libero_10_abs
```

**Arguments:**

| Argument | Required | Description |
|---|---|---|
| `--libero_task_suite` | Yes | Task suite: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90` |
| `--libero_raw_data_dir` | Yes | Path to directory containing raw `*_demo.hdf5` files |
| `--libero_target_dir` | Yes | Output directory for regenerated HDF5 files |
| `--resume` | No | Skip tasks whose output HDF5 already exists |

### Step 2: Convert to LeRobot format

```bash
python libero_h5.py \
    --src-paths ./data/processed/libero_10_abs \
    --output-path ./data/lerobot
```

**Arguments:**

| Argument | Required | Default | Description |
|---|---|---|---|
| `--src-paths` | Yes | — | One or more source directories containing regenerated HDF5 files |
| `--output-path` | Yes | — | Output directory for the LeRobot dataset |
| `--executor` | No | `local` | Execution backend: `local` or `ray` |
| `--workers` | No | `-1` (all CPUs) | Number of parallel workers |
| `--cpus-per-task` | No | `1` | CPUs per task (Ray only) |
| `--tasks-per-job` | No | `1` | Concurrent tasks per job (Ray only) |
| `--dataset-type` | No | `standard` | `standard` (axis-angle, 7-dim action) or `abs_quat` (quaternion, 8-dim action) |
| `--max-episodes-per-task` | No | all | Maximum episodes to convert per task |
| `--resume-from-save` | No | — | Logs directory to resume from the save step |
| `--resume-from-aggregate` | No | — | Logs directory to resume from the aggregate step |
| `--debug` | No | — | Run with 1 worker on 2 tasks, no hub push |
| `--push-to-hub` | No | — | Upload finished dataset to Hugging Face Hub |
| `--repo-id` | No | — | Hub repo ID (required with `--push-to-hub`) |
| `--push-only` | No | — | Skip conversion and push an existing dataset to hub |

### Dataset types

| Type | Action dimensions | Orientation |
|---|---|---|
| `standard` | 7 `[x, y, z, ax, ay, az, gripper]` | Axis-angle |
| `abs_quat` | 8 `[x, y, z, qx, qy, qz, qw, gripper]` | Quaternion |

### Combining multiple suites

Pass multiple source paths to aggregate them into a single LeRobot dataset:

```bash
python libero_h5.py \
    --src-paths ./data/processed/libero_spatial_abs ./data/processed/libero_object_abs \
    --output-path ./data/lerobot
```

### Upload to Hugging Face Hub

```bash
python libero_h5.py \
    --src-paths ./data/processed/libero_10_abs \
    --output-path ./data/lerobot \
    --push-to-hub \
    --repo-id your-username/libero-10-abs
```

To push an already-converted dataset without re-running conversion:

```bash
python libero_h5.py \
    --src-paths ./data/processed/libero_10_abs \
    --output-path ./data/lerobot \
    --push-only \
    --push-to-hub \
    --repo-id your-username/libero-10-abs
```

## Data Format

Raw LIBERO HDF5 files are expected at:
```
<libero_raw_data_dir>/<TASK_NAME>_demo.hdf5
```

Supported filename patterns:
- LIBERO scene format: `*_SCENE*_<task>_demo.hdf5`
- Generic LIBERO format: `<task>_demo.hdf5`
- MimicGen format: `<task>_d<num>.hdf5`

## Output Structure

After Step 1, regenerated HDF5 files are saved to `--libero_target_dir` alongside a `<suite>_metainfo.json` recording success/failure and initial states per episode.

After Step 2, the LeRobot dataset is written to:
```
<output-path>/<src-dir-name>_lerobot/      # single source
<output-path>/<src1>_<src2>_aggregated_lerobot/  # multiple sources
```



coffee_d2 
coffee_preparation_d1 
hammer_cleanup_d1 
kitchen_d1 
mug_cleanup_d1 
nut_assembly_d0 
pick_place_d0 
square_d2 
stack_d1 
stack_three_d1 
threading_d2 
three_piece_assembly_d2