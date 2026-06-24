# Dex-F3RM: Neural Feature Fields for Language-based Dexterous Robotic Manipulation

**Dex-F3RM** is a system for **dexterous grasping of unseen objects guided by open-text language
instructions**. It extends [F3RM](https://f3rm.github.io) (Feature Fields for Robotic Manipulation) from
parallel-jaw grippers to a 15-DOF anthropomorphic hand by distilling CLIP features into a 3D feature field and
representing each grasp with *per-link* local feature fields. Grasps are inferred few-shot from human
demonstrations and refined with a two-stage, language-guided optimization of both the 6-DOF palm pose and the
finger joint configuration.

> M.Sc. Thesis — [*Neural Feature Fields for Language-based Dexterous Robotic Manipulation*](https://doi.org/10.13140/RG.2.2.16378.04809)
> ([DOI: 10.13140/RG.2.2.16378.04809](https://doi.org/10.13140/RG.2.2.16378.04809))<br>
> **David Sebastian Martinez Lema** · Technical University of Munich (TUM), School of Computation, Information and
> Technology · M.Sc. Neuroengineering<br>
> Examiner: Prof. Alois Knoll · Supervisor: M.Sc. Qian Feng · Submitted 10.03.2025

[<img src="assets/images/f3rm_robot/optimize.gif" width="640" alt="Language-Guided Dexterous Grasp Optimization">](f3rm_robot/README.md)

-----

## Highlights

- **Dexterous grasp representation.** A grasp is defined by a 6-DOF palm pose `T ∈ SE(3)` *and* a hand joint
  configuration. Instead of a single cloud of query points around the palm (as in F3RM), Dex-F3RM places a set of
  canonical frames on *every link of every finger* and samples query points around each. Forward kinematics moves
  these query points as the joints change, so the task embedding captures the chosen grasp primitive and per-finger
  dexterity — not just the palm pose.
- **Grasp Primitive Library.** Demonstrations are organized into a hierarchy inspired by human grasp taxonomies,
  with five primitives: **cylindrical**, **hook**, **pinch**, **tripod**, and **lumbrical**. An LLM
  ([Llama 3](https://ai.meta.com/blog/meta-llama-3/)) groups demos and writes natural-language descriptions for each
  category, enabling hierarchical, language-driven demo retrieval.
- **Two-stage few-shot optimization.** The palm pose is optimized first, then the joint configuration in a separate
  stage, using a language-guidance-weighted cosine-similarity loss against the demonstration task embedding.
- **Real2Sim benchmark.** A real-to-sim evaluation pipeline ([SAM2](https://ai.meta.com/sam2/) for segmentation,
  [FoundationPose](https://github.com/NVlabs/FoundationPose) for 6-DOF pose) loads the predicted object-grasp pairs
  into [Isaac Sim](https://developer.nvidia.com/isaac-sim) (via the
  [MultiGripperGrasp](https://irvlutd.github.io/MultiGripperGrasp) pipeline) for large-scale, physically realistic
  grasp evaluation.

### Results (from the thesis)

**Simulation (Isaac Sim, fall-time success rate, 120 grasps):** Dex-F3RM surpasses F3RM by **+11.2%** and SparseDFF
by **+5.6%** for grasps held longer than 3 s.

| Method                       | > 3s     | > 2s | > 1s | > 0s (contact) |
|------------------------------|----------|------|------|----------------|
| Grasp Sampler                | 2.5      | 2.5  | 2.5  | 7.5            |
| SparseDFF                    | 25.0     | 25.0 | 25.0 | 53.3           |
| F3RM                         | 19.4     | 19.5 | 20.0 | 90.6           |
| **Dex-F3RM 2-Stage (ours)**  | **30.6** | 30.8 | 31.8 | **94.3**       |

**Real-world (50 grasps, 5 YCB objects):**

| Method            | Success Rate | Run Time |
|-------------------|--------------|----------|
| SparseDFF         | 54.0%        | 16 s     |
| F3RM              | 60.0%        | 5 min    |
| **Dex-F3RM 2-Stage** | **72.0%** | 5 min    |

**Ablations:** primitive joint initialization (30.6%) beats random (19.5%) and zero (13.8%); two-stage joint
optimization (30.6%) beats one-stage (27.9%) and none (23.9%); hierarchical demo retrieval (0.75 accuracy) beats
F3RM-style flat matching (0.625), reaching 1.00 when the grasp type is named in the query.

-----

## System Overview

Dex-F3RM spans three repositories:

| Repository | Role |
|------------|------|
| **`dex-f3rm`** (this repo) | Feature-field training, the dexterous grasp representation, the grasp primitive library, and the two-stage language-guided grasp optimizer (`f3rm-optimize`). |
| [`dex-f3rm-inference`](https://github.com/david-s-martinez/dex-f3rm-inference) | The real-robot side: hand-eye calibration, robot scanning / RGB-D capture, ROS2 + MoveIt motion planning, and grasp execution on the **Diana 7** arm with the **DLR/HIT Hand II**. |
| [`isaac_sim_grasping`](https://github.com/IRVLUTD/isaac_sim_grasping) | The MultiGripperGrasp toolkit, repurposed as the **Real2Sim benchmark** in Isaac Sim. Cloned locally at `isaac_sim_grasping/`. |

YCB object meshes/USDs used for the benchmark come from
[`gazebo-objects`](https://github.com/david-s-martinez/gazebo-objects) (expected at
`isaac_sim_grasping/gazebo-objects/objects_gazebo/ycb/`).

**Hardware (real-world setup):** Diana 7 (7-DOF arm), DLR/HIT Hand II (15-DOF, 20 actuated joints in code),
wrist-mounted RealSense D435 (eye-in-hand), NVIDIA RTX Ada 6000, Ubuntu 22.04, ROS2 + MoveIt + Octomap, impedance
control on the hand.

### Inference pipeline

```
                                  scan scene (50 RGB-D views, eye-in-hand)
                                              │
"Yellow Mustard"  ──► CLIP ──► Demo Retrieval ─┤   Train F3RM-DFF feature field (ns-train f3rm)
                       │      (Grasp Primitive  │              │
                       │         Library)       └──────────────┤
                       ▼                                        ▼
              Grasp Primitive Selection ──► Initial Proposals (marching-cubes + language masking)
                                                        │
                                          Stage 1: 6-DOF pose optimization
                                          Stage 2: joint configuration optimization
                                                        │
                            ┌───────────────────────────┴───────────────────────────┐
                            ▼                                                         ▼
                Real world: MoveIt motion plan + execute               Real2Sim: SAM2 + FoundationPose
                (dex-f3rm-inference)                                    ► Isaac Sim eval (isaac_sim_grasping)
```

A summary of the codebase layout is in [assets/code_structure.md](assets/code_structure.md). The new
dexterous-grasping logic lives in [`f3rm_robot/`](f3rm_robot): `task.py` (Task + Grasp Primitive Library),
`args.py` (optimizer config), `optimize.py` (proposal + two-stage optimization), `benchmark_data.py` (export
grasps to the Isaac Sim benchmark format), and `assets/hithand_*` (DLR/HIT Hand II URDF, meshes, and task
embeddings).

-----

## Installation

**Note:** this repo requires an NVIDIA GPU with CUDA 11.7+ for NeRF and feature field distillation.

#### 0. (Recommended) Build and run the Docker container with VS Code Dev Containers

The repository ships a [`.devcontainer`](.devcontainer) (`devcontainer.json` + Dockerfile) so you can build and run
the whole pipeline inside a container. You will need:

- Docker installed on your system
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Install the **Dev Containers** extension in VS Code, open this repository, and when prompted click **"Reopen in
Container"** (or run *"Dev Containers: Reopen in Container"* from the command palette, `Ctrl+Shift+P`). If this works,
you can skip steps 1–2 and run the remaining commands inside the container.

#### 1. Set up the conda environment

```bash
conda create -n f3rm python=3.8
conda activate f3rm
```

#### 2. Install Nerfstudio dependencies

```bash
# Install torch per https://pytorch.org/get-started/locally/ (CUDA 11.8 used here)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA toolkit (skip if you already have CUDA 11.8 — check with `nvcc --version`)
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX

# Install tiny-cuda-nn (takes a few minutes). See INSTALL.md if building from source.
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

#### 3. Install Dex-F3RM

```bash
# Install the package and its dependencies (if it fails, try: sudo chmod -R 777 ./)
pip install -e .

# For the correct Viser version
pip install viser==0.1.34

# Install command-line completions for nerfstudio
ns-install-cli

# Check that 'f3rm' is a valid method
ns-train --help
```

Tested on Nerfstudio 0.3.3 / 0.3.4. If you have a previous Nerfstudio install, run `which -a ns-train` and confirm
the first entry points to `$CONDA_PREFIX/bin/ns-train`.

#### 4. Install dependencies for the robot / grasp-optimization code

```bash
# Robot dependencies (open3d, params-proto, PyMCubes, viser, ...)
pip install -e ".[robot]"

# PyTorch3D (build from source). If it freezes:
#   export MAX_JOBS=4 && pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@stable" --user
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Sanity check — you should see a help message
f3rm-optimize --help
```

The Real2Sim benchmark additionally requires **Isaac Sim 2023.1.0** (see
[`isaac_sim_grasping`](https://github.com/IRVLUTD/isaac_sim_grasping)), and **SAM2** / **FoundationPose** for the
real-to-sim object segmentation and pose estimation steps.

-----

## Usage

### 1. Train a feature field (F3RM-DFF)

Scan a scene with a calibrated RGB-D camera, then train a NeRF that also distills dense CLIP features
(`ViT-L/14@336px`). On an RTX Ada 6000 this takes ~3 minutes for 5000 iterations.

```bash
# COLMAP-processed eye-in-hand scene
ns-train f3rm --max-num-iterations 5000 --output-dir f3rm_outputs --experiment-name scene_001 --timestamp '' \
  nerfstudio-data --data ./datasets/eyeinhand_nerf1/img_scene_001 --orientation-method none --auto-scale-poses True
```

To distill DINO features instead of CLIP, add `--pipeline.datamanager.feature-type DINO`. See `ns-train f3rm -h`
for all options, and the upstream [Nerfstudio docs](https://docs.nerf.studio/quickstart/custom_dataset.html) for
preparing your own datasets. You can inspect a trained field (PCA / similarity heatmaps) with
`ns-viewer --load-config f3rm_outputs/scene_001/f3rm/config.yml`.

### 2. Collect demonstrations and build task embeddings

Demonstrations are collected by tele-operating the hand (e.g. with a space mouse) to a grasp, recording the palm
pose and joint configuration, and labeling the grasp in natural language. Each demo's per-link query points are
transformed by forward kinematics, sampled against the feature field, and averaged into a **task embedding**.

The thesis uses **14 tasks** (object parts), **2 demos each**, across **10 objects**, covering the five grasp
primitives. Task embeddings ship pre-computed in
[`f3rm_robot/assets/hithand_tasks_og_fk/`](f3rm_robot/assets/hithand_tasks_og_fk) and are loaded by
`f3rm_robot/task.py::get_tasks`. The Grasp Primitive Library hierarchy is defined in
`task.py::grasp_primitives_dict`.

To regenerate task embeddings from demo scenes, see
[`f3rm_robot/assets/aquire_features.sh`](f3rm_robot/assets/aquire_features.sh), which calls:

```bash
python3 f3rm_robot/examples/generate_task.py \
  --scene f3rm_outputs/<demo_scene>/f3rm/config.yml \
  --demo_fname scene_demo_<name>.json --save --disable_visualize
```

### 3. Language-guided grasp optimization

```bash
f3rm-optimize --scene f3rm_outputs/scene_001/f3rm/config.yml
```

You will be prompted for an open-text query (e.g. *"hammer handle"*, *"red mug"*, *"yellow mustard bottle"*). The
optimizer then:

1. **Retrieves** the closest grasp primitive category and the best-matching demonstration(s) from the Grasp
   Primitive Library (hierarchical CLIP matching).
2. **Proposes** initial grasps by sampling a voxel grid (marching cubes), pruning free space by density, then
   keeping the top `--max-voxels` voxels by language similarity, and sampling `--num-rots-per-voxel` rotations.
3. **Optimizes** the palm pose (stage 1) and the joint configuration (stage 2) with Adam against the
   language-guidance-weighted cosine-similarity loss, pruning high-loss and colliding grasps.

Key flags (see `f3rm_robot/args.py` for all defaults):

| Flag | Default | Meaning |
|------|---------|---------|
| `--is-split-joint-optim` | `True` | Two-stage (pose then joints) optimization. |
| `--is-use-grasp-prompt` | `True` | Select the grasp primitive from the prompt (hierarchical retrieval). |
| `--num-steps` | `200` | Adam optimization steps. |
| `--lr-pose` / `--lr-joints` | `2e-3` / `2e-2` | Learning rates for the two stages. |
| `--softmax-temperature` | `0.001` | Temperature for language masking of voxels. |
| `--max-voxels` | `400` | Voxels kept after language similarity ranking. |
| `--num-rots-per-voxel` | `15` | Rotations sampled per voxel for initial proposals. |
| `--tasks-folder` | `hithand_tasks_og_fk` | Task-embedding set to load. |
| `--min-bounds` / `--max-bounds` | — | Workspace bounds in the world frame (metric). |
| `--visualize` | `True` | Viser visualizer at `http://localhost:8012`. |

The optimizer writes a ranked list of grasps to `grasps_to_world.pt` (palm poses) and `joints.pt` (joint configs)
under the feature field output directory. A detailed tutorial and FAQ for the optimizer is in
[`f3rm_robot/README.md`](f3rm_robot/README.md).

> **Note on collision checking:** `f3rm_robot/collision.py` ships with the upstream Panda gripper mesh. For the
> DLR/HIT Hand II, collision geometry is provided under `f3rm_robot/assets/hithand_palm/`.

### 4. Execute on the real robot

Loading the ranked grasps, running IK / MoveIt motion planning, and executing on the Diana 7 + DLR/HIT Hand II is
handled by the separate [`dex-f3rm-inference`](https://github.com/david-s-martinez/dex-f3rm-inference) repository
(hand-eye calibration, robot scanning, ROS2 control). This repo does not contain the real-robot drivers.

### 5. Real2Sim benchmark (Isaac Sim)

To evaluate grasps at scale without a robot:

```bash
# Train feature fields for the benchmark scenes (img_ycb_scene_1..5)
./get_benchmark_data.sh

# Run the two-stage optimizer over every benchmark scene; prompts for a model name
python run_benchmark.py
```

`run_benchmark.py` runs `f3rm-optimize` over `img_ycb_scene_1..5` and snapshots the `args.py` used for the run.
`f3rm_robot/benchmark_data.py` then converts the optimized grasps into the MultiGripperGrasp JSON format
(`hithand-<object>-<model>-<prompt>.json`), placing each YCB object at the pose recovered by **SAM2 +
FoundationPose** and pairing it with the predicted hand pose and joint configuration. These files are loaded into
**Isaac Sim** via [`isaac_sim_grasping`](https://github.com/IRVLUTD/isaac_sim_grasping), where many hands are
simulated in parallel and grasps are scored by **fall time** (gravity is enabled after ≥3 contact points;
> 3 s held = most successful).

The prompt-matching ablation data is in [`matching.csv`](matching.csv) (standard / hierarchical / grasp-type-in-prompt).

-----

## Troubleshooting

### Language queries are not registering in the Nerfstudio viewer
The viewer can fail to register text input if you reuse the same browser tab across runs, so `feature_pca` /
`similarity` may not appear in Render Options. Close and reopen the viewer tab.

### Running out of GPU memory
This codebase was tested on an RTX3090 (24 GB) and an RTX Ada 6000. Peak usage when training a CLIP feature field
without the viewer is ~6 GB, and ~12 GB with the viewer.
- Lower `--pipeline.model.eval-num-rays-per-chunk` (e.g. `8192`) during `ns-train`.
- Lower `--ray-samples-per-batch` (optimization, default `2**18`) and
  `--CollisionArgs.ray-samples-per-batch` (collision checking, default `2**22`) for `f3rm-optimize`.
- Decrease `Max Res` in the Nerfstudio viewer.

For more, see the upstream [F3RM robot README](f3rm_robot/README.md#troubleshooting).

-----

## Acknowledgements

Dex-F3RM builds directly on the open-source work of:

- [F3RM](https://github.com/f3rm/f3rm) — Feature Fields for Robotic Manipulation
- [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [Viser](https://github.com/nerfstudio-project/viser)
- [SparseDFF](https://arxiv.org/abs/2310.16838)
- [CLIP](https://github.com/openai/CLIP) / [MaskCLIP](https://github.com/chongzhou96/MaskCLIP),
  [DINO](https://github.com/facebookresearch/dino), [LERF](https://github.com/kerrj/lerf)
- [SAM2](https://github.com/facebookresearch/segment-anything-2) and
  [FoundationPose](https://github.com/NVlabs/FoundationPose)
- [MultiGripperGrasp / isaac_sim_grasping](https://github.com/IRVLUTD/isaac_sim_grasping) and
  [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [Llama 3](https://ai.meta.com/blog/meta-llama-3/) for building the grasp primitive hierarchy

With thanks to supervisor M.Sc. Qian Feng and Prof. Alois Knoll at the Chair of Robotics, Artificial Intelligence
and Real-time Systems, TUM.

## Citation

If you find this work useful, please cite the thesis and the original F3RM paper:

```bibtex
@mastersthesis{martinez2025dexf3rm,
    title  = {Neural Feature Fields for Language-based Dexterous Robotic Manipulation},
    author = {Martinez Lema, David Sebastian},
    school = {Technical University of Munich (TUM)},
    year   = {2025},
    type   = {Master's Thesis},
    doi    = {10.13140/RG.2.2.16378.04809},
    url    = {https://doi.org/10.13140/RG.2.2.16378.04809},
    note   = {Examiner: Prof. Alois Knoll; Supervisor: M.Sc. Qian Feng}
}

@inproceedings{shen2023F3RM,
    title     = {Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation},
    author    = {Shen, William and Yang, Ge and Yu, Alan and Wong, Jansen and Kaelbling, Leslie Pack and Isola, Phillip},
    booktitle = {7th Annual Conference on Robot Learning},
    year      = {2023},
    url       = {https://openreview.net/forum?id=Rb0nGIt_kh5}
}
```
