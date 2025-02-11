"""
Example script for generating a Task, which contains the query points and demo embeddings, given a feature field and
a set of demonstrations. We:

1. Load the feature field and demos
2. Sample a set of query points
3. Transform the query points by each demo pose
4. Sample the feature field for the density and feature at each transformed query point
5. Save it into a Task object

This script also supports visualization of the query points along with the demo gripper poses. Note that we save the
density not the alphas, as the voxel size can vary at optimization time.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple
from pynput import keyboard
import numpy as np
import open3d as o3d
import torch
from jaxtyping import Float
from pytorch3d.transforms import Transform3d, euler_angles_to_matrix
from scipy.spatial.transform import Rotation

from f3rm_robot.assets import get_panda_gripper_mesh, get_hithand_gripper_mesh, get_query_frames_fk, apply_joint_correction_torch
from f3rm_robot.load import LoadState, load_nerfstudio_outputs
from f3rm_robot.task import Task, sample_query_points
from f3rm_robot.visualizer import BaseVisualizer, ViserVisualizer

# Define translation and rotation step sizes
TRANSLATION_STEP = 0.001
ROTATION_STEP = torch.tensor([5.0, 5.0, 5.0]).mul(torch.pi / 180)  # Convert degrees to radians

# Key mapping for translation and rotation
TRANSLATION_KEYS = {
    "w": torch.tensor([TRANSLATION_STEP, 0, 0]),
    "s": torch.tensor([-TRANSLATION_STEP, 0, 0]),
    "a": torch.tensor([0, -TRANSLATION_STEP, 0]),
    "d": torch.tensor([0, TRANSLATION_STEP, 0]),
    "W": torch.tensor([0, 0, TRANSLATION_STEP]),
    "S": torch.tensor([0, 0, -TRANSLATION_STEP]),
}

ROTATION_KEYS = {
    "i": torch.tensor([ROTATION_STEP[0], 0, 0]),  # Rotate +X
    "k": torch.tensor([-ROTATION_STEP[0], 0, 0]),  # Rotate -X
    "j": torch.tensor([0, -ROTATION_STEP[1], 0]),  # Rotate -Y
    "l": torch.tensor([0, ROTATION_STEP[1], 0]),  # Rotate +Y
    "I": torch.tensor([0, 0, ROTATION_STEP[2]]),  # Rotate +Z
    "K": torch.tensor([0, 0, -ROTATION_STEP[2]]),  # Rotate -Z
}

def modify_demo_pose(visualizer, demo_dict,load_state, demo_poses, joints_torch, torques_torch, is_finger_qp, num_query_points, num_fingers, num_finger_qf, qp_std_dev, index, device):
    """Modify the selected demo pose using keyboard inputs."""
    pose = demo_poses.get_matrix().clone().transpose(1,2)

    print("\nUse WASD for XY translation, W/S (caps) for Z translation")
    print("Use IJKL for rotation, I/K (caps) for Z rotation")
    print("Press ENTER to confirm and exit.")

    is_running = True  # Flag to control loop
    movement = {"translation": torch.zeros(3), "rotation": torch.zeros(3)}

    def on_press(key):
        """Handles key press events for modifying pose."""
        nonlocal is_running
        
        try:
            key_str = key.char  # Convert key object to string
            
            # Apply translation
            if key_str in TRANSLATION_KEYS:
                movement["translation"] += TRANSLATION_KEYS[key_str]
            
            # Apply rotation
            elif key_str in ROTATION_KEYS:
                movement["rotation"] += ROTATION_KEYS[key_str]

            # Confirm and exit
            elif key_str == "\r":  # Enter key
                print("Pose updated!")
                is_running = False

        except AttributeError:
            pass  # Ignore special keys

    def on_release(key):
        """Handles key release events (optional)."""
        nonlocal is_running
        if key == keyboard.Key.esc:  # Escape key cancels modification
            print("Pose modification cancelled.")
            is_running = False
            return False  # Stop listener

    # Start listening for keyboard input
    with keyboard.Listener(on_press=on_press, on_release=on_release, suppress=True) as listener:
        while is_running:
            if torch.any(movement["translation"] != 0):
                pose[index,:3,-1]+= movement["translation"]
                movement["translation"].zero_()  # Reset translation after applying

            if torch.any(movement["rotation"] != 0):
                R = euler_angles_to_matrix(movement["rotation"], "XYZ")
                pose[index,:3, :3] = R @ pose[index,:3, :3]
                movement["rotation"].zero_()  # Reset rotation after applying

            demo_poses = Transform3d(matrix=pose.transpose(1,2))
            time.sleep(0.1)  # Small delay for better user experience
            print("\nUpdated Pose:")
            print(demo_poses[index].get_matrix().cpu().numpy())
            result = get_demo_qp(
            load_state, 
            demo_poses, 
            joints_torch, 
            torques_torch, 
            is_finger_qp, 
            num_query_points, 
            num_fingers, 
            num_finger_qf, 
            qp_std_dev, 
            device
        )
            outputs, query_points, qp_transformed, link_points, demo_poses, joints_torch, joints_np, torques_torch = result 
            visualize_demos(
                visualizer, 
                load_state, 
                demo_labels = demo_dict["demo_labels"], 
                demo_poses = demo_poses, 
                demo_qps = qp_transformed, 
                demo_joints = joints_np,
                is_update_pcd=False,
                is_update_qp=False
            )
    
    return outputs, query_points, qp_transformed, link_points, demo_poses, joints_torch, joints_np, torques_torch


def load_demos(demo_path: Path) -> Tuple[Dict, Transform3d]:
    """Load the demos from the given demo JSON file."""
    if not demo_path.exists():
        raise FileNotFoundError(f"Could not find demo file at {demo_path}.")
    with open(demo_path, "r") as f:
        demo_dict = json.load(f)

    if len(demo_dict["demo_labels"]) != len(demo_dict["demo_poses"]):
        raise ValueError("demo_labels and demo_poses must have the same length.")

    # Load demo poses
    transforms = []
    joints = []
    torques = []
    for label, pose in zip(demo_dict["demo_labels"], demo_dict["demo_poses"]):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(pose["quat_xyzw"]).as_matrix()
        transform[:3, 3] = pose["translation"]
        transform = torch.from_numpy(transform).float()
        joints_torch = torch.FloatTensor(pose["joint_state"]).float()
        torques_torch = torch.FloatTensor(pose["torque_state"]).float()
        # Need to take transpose as Transform3d uses row instead of column vectors
        transform = Transform3d(matrix=transform.T)
        transforms.append(transform)
        joints.append(joints_torch)
        torques.append(torques_torch)

    transforms = Transform3d.stack(*transforms)
    joints = torch.stack(joints)
    torques = torch.stack(torques)
    return demo_dict, transforms, joints, torques


def visualize_demos(
    visualizer: BaseVisualizer,
    load_state: LoadState,
    demo_labels: List[str],
    demo_poses: Transform3d,
    demo_qps: Float[torch.Tensor, "d n 3"],
    demo_joints: np.ndarray,
    is_update_pcd: bool = True,
    is_update_qp: bool = True
):
    if is_update_pcd:
        # We use helpers from the main optimization scripts for visualization
        from f3rm_robot.optimize import get_scene_pcd

        # Show point cloud of the scene
        scene_pcd = get_scene_pcd(load_state, num_points=100_000, voxel_size=0.005)
        visualizer.add_o3d_point_cloud("scene_pcd", scene_pcd, point_size=0.005 + 0.001)
    # Show query points, coordinate frame, and gripper mesh for each demo
    # base_gripper_mesh = get_panda_gripper_mesh()
    try:
        joints = [demo for demo in demo_joints]
    except:
        print("Joints not provided.")
        joints = [np.zeros((5,4)) for demo in demo_joints]

    for label, qps, pose, joint in zip(demo_labels, demo_qps, demo_poses, joints):
        base_gripper_mesh, f3rm_tfs = get_hithand_gripper_mesh(np.array(joint), is_use_coll_mesh = True, is_get_tf_dict = True)
        base_gripper_mesh.compute_vertex_normals()
        # Transformation matrix, need to transpose Transform3d matrix as it uses row vectors
        transform = pose.get_matrix()[0].T
        if is_update_qp:
            # Transformed query points
            qp_pcd = o3d.geometry.PointCloud()
            qp_pcd.points = o3d.utility.Vector3dVector(qps.cpu().numpy())
            qp_pcd.paint_uniform_color([1, 0, 0])
            visualizer.add_o3d_point_cloud(f"{label}/qp", qp_pcd, point_size=0.005)

        # Coordinate frame for gripper pose
        if is_update_qp:
            pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            pose_frame.transform(transform)
            visualizer.add_o3d_mesh(f"{label}/frame", pose_frame)
        if is_update_qp:
            # Coordinate frame for fingers pose
            for i, (name, tf) in enumerate(f3rm_tfs.items()):
                pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
                pose_frame.transform(tf).transform(transform)
                visualizer.add_o3d_mesh(f"{label}/{i}", pose_frame)

        # Gripper mesh
        gripper_mesh = o3d.geometry.TriangleMesh(base_gripper_mesh)
        gripper_mesh.transform(transform)
        visualizer.add_o3d_mesh(f"{label}/gripper_mesh", gripper_mesh)

def get_demo_qp(load_state, demo_poses, joints_torch, torques_torch, is_finger_qp, num_query_points, num_fingers, num_finger_qf, qp_std_dev, device):
    # Sample query points and transform by demo poses. The optimization can be noisy depending on the sampling of the
    # query points, so you might want to try multiple different samples.
    joints_np = joints_torch.detach().cpu().numpy().reshape(-1,5,4)

    if is_finger_qp:
        tot_num_qp = int(num_query_points/(num_fingers * num_finger_qf + 1))
        link_points = sample_query_points(tot_num_qp, mean=(0.0,0.0,0.0), std_dev=qp_std_dev)
        query_points_og = get_query_frames_fk(np.zeros((5,4))).transform_points(link_points)
        query_points = torch.stack([get_query_frames_fk(joint).transform_points(link_points) for joint in joints_np])
        n, j, q, d = query_points.shape
        query_points = query_points.view(n, j * q, d) # n_demo, n_joints, n_query points, dim_query points
        qp_transformed = demo_poses.transform_points(query_points)
        # TODO: investigate what is the effect of different query points in Task
        # avg qp, 
        # vs selecting one of them randomly, 
        # vs using set of finger qp before fk, 
        # vs use og qp & finger qp separately
        query_points = query_points_og.view(j * q, d)
        # query_points = query_points.mean(dim=0) #better, results look similar to og qp approach, with some collisions, weird orientations, good prompts help
        # query_points = query_points[random.randint(0, n-1)] #really bad, poses are not aligned with objects
    else:
        query_points = sample_query_points(100, mean=(0.025,0.0,0.0), std_dev=0.025)
        link_points = None
        qp_transformed = demo_poses.transform_points(query_points)
    qp_transformed = qp_transformed.to(device)

    # Get features and density for each demo from feature field
    feature_field = load_state.feature_field_adapter()
    with torch.no_grad():
        outputs = feature_field(qp_transformed)
        # Visualize the scene and demos if required
    return outputs, query_points, qp_transformed, link_points, demo_poses, joints_torch, joints_np, torques_torch

def generate_task(
    scene: str,
    demo_fname: str,
    num_query_points: int,
    qp_std_dev: float,
    save: bool,
    disable_visualize: bool,
    viser_host: str,
    viser_port: int,
    num_fingers: int = 5,
    num_finger_qf: int = 3,
    is_finger_qp: bool = True,
    is_fix_demos: bool = False,
):
    """Generate Task for the given scene and demos."""
    # Load the feature field
    load_state = load_nerfstudio_outputs(scene)
    device = load_state.pipeline.device

    # Load the demos for this scene
    dataset = load_state.pipeline.datamanager.get_datapath()
    demo_dict, demo_poses, joints_torch, torques_torch = load_demos(dataset / demo_fname)
    joints_torch = joints_torch.to(device)
    joints_torch = apply_joint_correction_torch(joints_torch)
    torques_torch = torques_torch.to(device)
    joints_np = joints_torch.detach().cpu().numpy().reshape(-1,5,4)

    result = get_demo_qp(
        load_state, 
        demo_poses, 
        joints_torch, 
        torques_torch, 
        is_finger_qp, 
        num_query_points, 
        num_fingers, 
        num_finger_qf, 
        qp_std_dev, 
        device
    )
    outputs, query_points, qp_transformed, link_points, demo_poses, joints_torch, joints_np, torques_torch = result
    if not disable_visualize:
        visualizer = ViserVisualizer(host=viser_host, port=viser_port)
        visualize_demos(
            visualizer, 
            load_state, 
            demo_labels = demo_dict["demo_labels"], 
            demo_poses = demo_poses, 
            demo_qps = qp_transformed, 
            demo_joints = joints_np,
            is_update_qp = not is_fix_demos
        )
        try:
            input("Press Enter or Ctrl+C to exit.")
        except KeyboardInterrupt:
            print()
            pass
        print("Exiting...")

    is_end = "n"   
    while is_end == "n" and is_fix_demos:
        print(f"Available poses: 0 to {len(demo_poses) - 1}")
        index = int(input("Enter index of pose to modify: "))

        result = modify_demo_pose(
            visualizer, 
            demo_dict,
            load_state,
            demo_poses,
            joints_torch,
            torques_torch,is_finger_qp,
            num_query_points,num_fingers,
            num_finger_qf,
            qp_std_dev,
            index,
            device
        )

        outputs, query_points, qp_transformed, link_points, demo_poses, joints_torch, joints_np, torques_torch = result
        if is_fix_demos:
            is_end = str(input("End y/n: "))
        else:
            is_end = "y"
        
    # Create task and save
    task = Task(
        name = demo_dict["task"],
        query_points = query_points,
        link_points = link_points,
        demo_features = outputs["feature"],
        demo_density = outputs["density"],
        demo_joints = joints_torch,
        demo_torques = torques_torch,  
    )
    print(f"Created task '{task.name}' with {task.num_demos} demos and {task.num_query_points} query points.")
    if save:
        if is_finger_qp:
            save_path = f"f3rm_robot/assets/hithand_tasks_og_fk/{task.name}.pt"
        else:
            save_path = f"f3rm_robot/assets/hithand_tasks_og/{task.name}.pt"
        torch.save(task, save_path)
        print(f"Saved task to {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Example script to generate a Task for a feature field and demos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene", type=str, required=True, help="Path to Nerfstudio scene config.yml file for the f3rm training run."
    )
    parser.add_argument("--demo_fname", type=str, default="scene_demo.json", help="Name of the demo file.")
    parser.add_argument("--num_query_points", type=int, default=320, help="Number of query points to sample.")
    parser.add_argument("--qp_std_dev", type=float, default=0.004, help="Standard deviation of query points.")
    # parser.add_argument("--qp_std_dev", type=float, default=0.02, help="Standard deviation of query points.")
    parser.add_argument("--save", action="store_true", help="Save the task to disk under the task name.")
    parser.add_argument("--disable_visualize", action="store_true", help="Disable visualization of the task.")
    parser.add_argument("--viser_host", type=str, default="localhost", help="Host for Viser Visualizer.")
    parser.add_argument("--viser_port", type=int, default=8012, help="Port for Viser Visualizer.")

    args = parser.parse_args()
    generate_task(**vars(args))
