import os
from typing import Tuple

from params_proto import ParamsProto, PrefixProto, Proto

class OptimizationArgs(ParamsProto, cli_parse=False):
    """
    Language-Guided 6-DOF Pose Optimization for a given scene.
    """

    scene: str = Proto(help="Path to Nerfstudio scene config.yml file for the f3rm training run.")
    benchmarks: list = Proto([f"img_ycb_scene_{i}" for i in range(6)], help="Benchmark scenes.")
    model_name: str = Proto("og-f3rm-paper_2025-02-27_150214", help="Benchmark scenes")
    benchmark_path: str = Proto("datasets/eyeinhand_nerf1", help="Path to bechmark folder.")
    benchmark_config: str = Proto("scene_benchmark_data_better.json", help="Path to bechmark config.")
    is_prompt_match_bench: bool = Proto(False, help="Whether to run benchmark for prompts")
    
    # benchmark_config: str = Proto("scene_benchmark_data.json", help="Path to bechmark config.")
    # Initial proposals
    tasks_folder: str = Proto("hithand_tasks_og_fk", help="Name of tasks folder.")
    # tasks_folder: str = Proto("hithand_tasks_avg_fk", help="Name of tasks folder.")
    # tasks_folder: str = Proto("hithand_tasks_og", help="Name of tasks folder.")
    is_use_grasp_prompt: bool = Proto(True, help="Whether to enable selection of grasp primitive from prompt.")
    voxel_size: float = Proto(0.01, help="Voxel size to discretize workspace into (in meters).")
    num_rots_per_voxel: int = Proto(15, help="Number of rotations to sample for each voxel.")
    # num_rots_per_voxel: int = Proto(4, help="Number of rotations to sample for each voxel.")
    alpha_threshold: float = Proto(0.1, help="Alpha threshold to use for marching cubes masking.")
    softmax_temperature: float = Proto(0.001, help="Temperature to use for softmax for language masking.")
    max_voxels: int = Proto(400, help="Max number of voxels after similarity with prompt.")

    # Optimization
    is_split_joint_optim: bool = Proto(True, help="Whether to optimize joints in a second stage.")
    num_steps: int = Proto(200, help="Number of optimization steps to use.")
    lr_pose: float = Proto(2e-3, help="Learning rate to use for language-guided pose optimization.")
    lr_joints: float = Proto(2e-2, help="Learning rate to use for language-guided joint optimization.")
    ray_samples_per_batch: int = Proto(
        2**18, help="Number of ray samples to use per batch. Decrease if you are running out of CUDA memory."
    )

    # Pruning
    keep_proportion: float = Proto(
        0.975, help="Proportion of proposals to keep after pruning for each optimization step."
    )
    min_proposals: int = Proto(2048, help="Minimum number of proposals to keep after pruning.")
    prune_after: int = Proto(10, help="Number of optimization steps to run before pruning.")
    num_outs: int = Proto(50, help="Max number of outputs (for benchmark).")
    
    # Min and max bounds of the workspace in world frame with metric scale
    min_bounds: Tuple[float, float, float] = (-0.2, -0.80, 0.07)
    max_bounds: Tuple[float, float, float] = (0.7, 0.25, 0.30)

    # Visualization
    visualize: bool = Proto(True, help="Whether to enable visualization of the optimization. This slows down the run.")
    viser_host: str = Proto("localhost", help="Host to use for viser visualization server.")
    viser_port: int = Proto(8012, help="Port to use for viser visualization server.")
    num_poses_to_visualize: int = Proto(10, help="Number of poses to visualize during and after optimization.")



class CollisionArgs(PrefixProto, cli_parse=False):
    """Arguments for collision checking a proposed grasp. The default values work well for the default Panda gripper."""

    alpha_threshold: float = Proto(0.2, help="Alpha threshold for a point to be considered to be occupied.")
    voxel_size: float = Proto(
        0.0075,
        help="Voxel size to voxelize the Panda gripper. You may need to adjust alpha if you change this.",
    )
    overlap_num: int = Proto(10, help="Number of intitial overlapping points to be considered a collision.")
    overlap_num_final: int = Proto(10, help="Number of final overlapping points to be considered a collision.")

    allow_finger_collisions: bool = Proto(
        False,
        help="Whether to allow collisions between the fingers, and hence use the gripper model without the fingers.",
    )
    ray_samples_per_batch: int = Proto(
        2**22,
        help="Number of ray samples to use per batch for collision checking, decrease if running out of memory.",
    )

# You can access the variables directly with OptimizationArgs.<field_name>, and do not need to instantiate an object
# of this class.
_args = OptimizationArgs


def validate_args():
    assert _args.scene, "Must specify scene config file using --scene."
    assert os.path.exists(_args.scene), f"--scene config file {_args.scene} does not exist"
    
    # Initial proposals
    assert 0 < _args.voxel_size < 0.1, f"--voxel_size should be between 0 and 0.1"
    assert _args.num_rots_per_voxel > 0, "--num_rots_per_voxel must be positive"
    assert 0 < _args.alpha_threshold <= 1.0, "--alpha_threshold must be between 0 and 1"
    assert _args.softmax_temperature > 0, "--softmax_temperature must be positive"
    # Optimization
    assert _args.num_steps > 0, "--num_steps must be positive"
    assert _args.lr_pose > 0, "--lr must be positive"
    assert _args.lr_joints > 0, "--lr must be positive"
    assert _args.ray_samples_per_batch > 0, "--ray_samples_per_batch must be positive"
    # Pruning
    assert 0 < _args.keep_proportion <= 1.0, "--keep_proportion must be between 0 and 1"
    assert _args.min_proposals > 0, "--min_proposals must be positive"
    assert _args.prune_after > 0, "--prune_after must be positive"
    # Check min and max bounds
    assert len(_args.min_bounds) == 3, f"--min_bounds must be a tuple of length 3, not {_args.min_bounds}"
    assert len(_args.max_bounds) == 3, f"--max_bounds must be a tuple of length 3, not {_args.max_bounds}"
    assert all(
        [min_bound < max_bound for min_bound, max_bound in zip(_args.min_bounds, _args.max_bounds)]
    ), "--min_bounds must be less than --max_bounds"
    # Visualization - try process args.visualize
    if isinstance(_args.visualize, str):
        assert _args.visualize.lower() in {"true", "false"}, "--visualize must be True or False"
        _args.visualize = _args.visualize.lower() == "true"
    assert _args.viser_port > 0, "--viser_port must be positive"
