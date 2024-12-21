import os

import open3d as o3d
import numpy as np
import urdfpy
_MODULE_PATH = os.path.dirname(__file__)


def get_asset_path(asset_name: str) -> str:
    return os.path.join(_MODULE_PATH, asset_name)

HAND_CFG = {
    'Right_Index_0': 0.2,
    'Right_Index_1': 0.2,
    'Right_Index_2': 0.2,
    'Right_Index_3': 0.2,
    'Right_Little_0': 0.2,
    'Right_Little_1': 0.2,
    'Right_Little_2': 0.2,
    'Right_Little_3': 0.2,
    'Right_Middle_0': 0.2,
    'Right_Middle_1': 0.2,
    'Right_Middle_2': 0.2,
    'Right_Middle_3': 0.2,
    'Right_Ring_0': 0.2,
    'Right_Ring_1': 0.2,
    'Right_Ring_2': 0.2,
    'Right_Ring_3': 0.2,
    'Right_Thumb_0': 0.2,
    'Right_Thumb_1': 0.2,
    'Right_Thumb_2': 0.2,
    'Right_Thumb_3': 0.2,
}

def get_hand_cfg_map(cfg_arr):
    cfg_map = HAND_CFG
    keys = sorted(HAND_CFG.keys())
    for idx, k in enumerate(keys):
        cfg_map[k] = cfg_arr[idx]
    return cfg_map

def full_joint_conf_from_partial_joint_conf(partial_joint_conf):
    """Takes in the 15 dimensional joint conf output from VAE and repeats the 3*N-th dimension to turn dim 15 into dim 20.

    Args:
        partial_joint_conf (np.array): Output from vae with dim(partial_joint_conf.position) = 15

    Returns:
        full_joint_conf (np.array): Full joint state with dim(full_joint_conf.position) = 20
    """
    full_joint_pos = 20 * [0]
    ix_full_joint_pos = 0
    for i, val in enumerate(partial_joint_conf):
        if (i + 1) % 3 == 0:
            full_joint_pos[ix_full_joint_pos] = val
            full_joint_pos[ix_full_joint_pos + 1] = val
            ix_full_joint_pos += 2
        else:
            full_joint_pos[ix_full_joint_pos] = val
            ix_full_joint_pos += 1

    full_joint_conf = full_joint_pos
    return full_joint_conf

def get_robot_fk(robot, joint_conf):
    # get the full joint config
    if joint_conf.shape[0] == 15:
        joint_conf_full = full_joint_conf_from_partial_joint_conf(joint_conf)
    elif joint_conf.shape[0] == 20:
        joint_conf_full = joint_conf
    else:
        raise Exception('Joint_conf has the wrong size in dimension one: %d. Should be 15 or 20' %
                        joint_conf.shape[0])
    cfg_map = get_hand_cfg_map(joint_conf_full)
    fk = robot.visual_trimesh_fk(cfg=cfg_map)

    return fk, robot.link_fk(cfg=cfg_map)

def get_panda_gripper_mesh() -> o3d.geometry.TriangleMesh:
    asset_path = get_asset_path("panda_gripper_visual.obj")
    return o3d.io.read_triangle_mesh(asset_path)

# TODO: use IK instead of fixed tf matrix
wrist2grasp = np.array([
    [-0.000,  1.000, -0.000,  0.045],
    [1.000,  0.000, -0.000,  0.000],
    [0.000, -0.000, -1.000,  0.150],
    [0.000,  0.000,  0.000,  1.000],
]) # computed using ros2 run tf2_ros tf2_echo base_link_hithand f3rm_link

def get_hithand_gripper_mesh(joint) -> o3d.geometry.TriangleMesh:
    asset_path = get_asset_path("hithand_palm/hithand.urdf")
    robot = urdfpy.URDF.load(asset_path)

    f3rm_names = {link.name:i for i, link in enumerate(robot.links) if "f3rm" in link.name}
    print(f"names: {f3rm_names}")

    fk, fk_link = get_robot_fk(robot, joint)

    f3rm_frames = {f"wrist_2_{name}": np.linalg.inv(wrist2grasp) @ fk_link[robot.links[idx]] for name, idx in f3rm_names.items()}

    mesh_robot_total = o3d.geometry.TriangleMesh()
    for tm in fk:
        pose = fk[tm]
        pose = np.linalg.inv(wrist2grasp) @ pose
        mesh_robot = o3d.geometry.TriangleMesh()
        mesh_robot.vertices = o3d.pybind.utility.Vector3dVector(np.asarray(tm.vertices.copy()))
        mesh_robot.triangles = o3d.pybind.utility.Vector3iVector(np.asarray(tm.faces.copy()))
        mesh_robot.transform(pose)
        mesh_robot_total += mesh_robot
    return mesh_robot_total, f3rm_frames
