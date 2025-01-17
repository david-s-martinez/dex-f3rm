import os
import torch
import open3d as o3d
import numpy as np
import urdfpy
from pytorch3d.transforms import Transform3d
from collections import OrderedDict
from urdfpy import URDF
import pytorch_kinematics as pk

def collision_trimesh_fk_lfk(robot, cfg=None, links=None):
    """Computes the poses of the URDF's collision trimeshes using fk.

    Parameters
    ----------
    cfg : dict or (n), float
        A map from joints or joint names to configuration values for
        each joint, or a list containing a value for each actuated joint
        in sorted order from the base link.
        If not specified, all joints are assumed to be in their default
        configurations.
    links : list of str or list of :class:`.Link`
        The links or names of links to perform forward kinematics on.
        Only trimeshes from these links will be in the returned map.
        If not specified, all links are returned.

    Returns
    -------
    fk : dict
        A map from :class:`~trimesh.base.Trimesh` objects that are
        part of the collision geometry of the specified links to the
        4x4 homogenous transform matrices that position them relative
        to the base link's frame.
    """
    lfk = robot.link_fk(cfg=cfg, links=links)

    fk = OrderedDict()
    for link in lfk:
        pose = lfk[link]
        cm = link.collision_mesh
        if cm is not None:
            fk[cm] = pose
    return fk, lfk

def visual_trimesh_fk_lfk(robot, cfg=None, links=None):
    """Computes the poses of the URDF's visual trimeshes using fk.

    Parameters
    ----------
    cfg : dict or (n), float
        A map from joints or joint names to configuration values for
        each joint, or a list containing a value for each actuated joint
        in sorted order from the base link.
        If not specified, all joints are assumed to be in their default
        configurations.
    links : list of str or list of :class:`.Link`
        The links or names of links to perform forward kinematics on.
        Only trimeshes from these links will be in the returned map.
        If not specified, all links are returned.

    Returns
    -------
    fk : dict
        A map from :class:`~trimesh.base.Trimesh` objects that are
        part of the visual geometry of the specified links to the
        4x4 homogenous transform matrices that position them relative
        to the base link's frame.
    """
    lfk = robot.link_fk(cfg=cfg, links=links)

    fk = OrderedDict()
    for link in lfk:
        for visual in link.visuals:
            for mesh in visual.geometry.meshes:
                pose = lfk[link].dot(visual.origin)
                if visual.geometry.mesh is not None:
                    if visual.geometry.mesh.scale is not None:
                        S = np.eye(4, dtype=np.float64)
                        S[:3,:3] = np.diag(visual.geometry.mesh.scale)
                        pose = pose.dot(S)
                fk[mesh] = pose
    return fk, lfk
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

def get_robot_fk(robot, joint_conf, is_fk_mesh = True):
    # get the full joint config
    if joint_conf.shape[0] == 15:
        joint_conf_full = full_joint_conf_from_partial_joint_conf(joint_conf)
    elif joint_conf.shape[0] == 20:
        joint_conf_full = joint_conf
    else:
        raise Exception('Joint_conf has the wrong size in dimension one: %d. Should be 15 or 20' %
                        joint_conf.shape[0])
    cfg_map = get_hand_cfg_map(joint_conf_full)

    if is_fk_mesh:
        fk, lfk = visual_trimesh_fk_lfk(robot, cfg=cfg_map)
        return fk, lfk
    else:
        lfk  = robot.link_fk(cfg=cfg_map)
        return None, lfk

def get_panda_gripper_mesh() -> o3d.geometry.TriangleMesh:
    asset_path = get_asset_path("panda_gripper_visual.obj")
    return o3d.io.read_triangle_mesh(asset_path)

# TODO: use IK instead of fixed tf matrix
wrist_T_grasp = np.array([
    [-0.000,  1.000, -0.000,  0.045],
    [1.000,  0.000, -0.000,  0.000],
    [0.000, -0.000, -1.000,  0.150],
    [0.000,  0.000,  0.000,  1.000],
]) # computed using ros2 run tf2_ros tf2_echo base_link_hithand f3rm_link (grasp in wrist frame) wrist_T_grasp | grasp2wrist
grasp_T_wrist = np.linalg.inv(wrist_T_grasp)
asset_path = get_asset_path("hithand_palm/hithand.urdf")
asset_path_coll = get_asset_path("hithand_palm/hithand_collision.urdf")
robot_og = URDF.load(asset_path)
robot_coll = URDF.load(asset_path_coll)
f3rm_names = {link.name:i for i, link in enumerate(robot_coll.links) if "f3rm" in link.name}
# Convert to PyTorch tensors
wrist_T_grasp_tensor = torch.from_numpy(wrist_T_grasp).float()
grasp_T_wrist_tensor = torch.from_numpy(grasp_T_wrist).float()

# Using pytorch_kinematics (loading both for collision and visualization)
chain_og = pk.build_chain_from_urdf(open(asset_path, mode="rb").read())
chain_coll = pk.build_chain_from_urdf(open(asset_path_coll, mode="rb").read())

def get_query_frames_fk_torch(joint, is_debug=False, is_return_tensor=True):
    """
    Computes forward kinematics for specified finger root links.

    Args:
        joint (torch.Tensor): Tensor of joint angles with shape (..., 20).
        is_debug (bool): Whether to print debug information.
        is_return_tensor (bool): Whether to return a single tensor or a dictionary of tensors.

    Returns:
        If is_return_tensor is True:
            torch.Tensor: A tensor of transforms with shape (..., num_f3rm_links, 4, 4).
        If is_return_tensor is False:
            dict: A dictionary mapping finger root link names to their 4x4 transformation matrices.
    """
    joint = apply_joint_correction_torch(joint)
    transform_list = []
    fk_links = chain_coll.to(device=joint.device).forward_kinematics(joint)
    transform_dict= {name:torch.transpose(grasp_T_wrist_tensor.to(joint.device) @ fk_links[name].get_matrix(), 1, 2) for name in f3rm_names.keys()}
    if is_debug:
        print(f"names: {transform_dict.keys()}")
    transform_list = torch.stack(list(transform_dict.values())).float()
    f3rm_frames = torch.transpose(transform_list, 0, 1)
    if is_return_tensor:
        return f3rm_frames
    else:
        return transform_dict

def get_query_frames_fk(joint, is_debug = False, is_return_tensor = True):
    # Reorder and apply correction
    joint = apply_joint_correction(joint)
    if is_debug:
        print(f"names: {f3rm_names}")
    _, fk_link = get_robot_fk(robot_coll, joint, False)
    if is_return_tensor:
        f3rm_frames = Transform3d(matrix=torch.from_numpy(np.stack([(grasp_T_wrist @ fk_link[robot_coll.links[idx]]).T for name, idx in f3rm_names.items()])).float())
    else:
        f3rm_frames = {f"grasp_T_{name}": grasp_T_wrist @ fk_link[robot_coll.links[idx]] for name, idx in f3rm_names.items()}
    return f3rm_frames

def apply_joint_correction_torch(joint):
    """
    Reorders and applies correction to joint angles.

    Args:
        joint (torch.Tensor): Tensor of joint angles with shape (..., 5, 4).

    Returns:
        torch.Tensor: Corrected joint angles flattened to shape (..., 20).
    """
    finger_order = {
        'Index': 1,
        'Little': 4,
        'Middle': 2,
        'Ring': 3,
        'Thumb': 0,
    }
    joint_correction = torch.tensor([[-1, 1, 1, 1]], dtype=joint.dtype, device=joint.device) # (1, 4)
    indices = torch.tensor(list(finger_order.values()), dtype=torch.long, device=joint.device) # (5,)
    
    # Correct shape for proper indexing and broadcasting
    return (joint[..., indices, :] * joint_correction).flatten(start_dim=-2)

def apply_joint_correction(joint):
    # joint->(5,4) (5 fingers, 4 joints)
    # The dict numbers represent the ordering of the fingers comming from the hand server.
    finger_order = {
    'Index': 1,
    'Little': 4,
    'Middle': 2,
    'Ring': 3,
    'Thumb': 0,
    }
    # invert the sign of the first joint to match real hand.
    joint_correction = np.ones(4)
    joint_correction[0]*= -1
    indices = np.array(list(finger_order.values()))
    return (joint[indices, :] * joint_correction[None, :]).flatten()

def get_hithand_gripper_mesh(joint = np.zeros((5,4)), is_debug = False, is_use_coll_mesh = False, robot = robot_og) -> o3d.geometry.TriangleMesh:
    # Reorder and apply correction
    joint = apply_joint_correction(joint)
    if is_debug:
        print(f"names: {f3rm_names}")
    if is_use_coll_mesh:
        robot = robot_coll
    fk, fk_link = get_robot_fk(robot, joint)

    f3rm_frames = {f"grasp_T_{name}": grasp_T_wrist @ fk_link[robot.links[idx]] for name, idx in f3rm_names.items()}

    mesh_robot_total = o3d.geometry.TriangleMesh()
    for tm in fk:
        pose = fk[tm]
        pose = grasp_T_wrist @ pose
        mesh_robot = o3d.geometry.TriangleMesh()
        mesh_robot.vertices = o3d.pybind.utility.Vector3dVector(np.asarray(tm.vertices.copy()))
        mesh_robot.triangles = o3d.pybind.utility.Vector3iVector(np.asarray(tm.faces.copy()))
        mesh_robot.transform(pose)
        mesh_robot_total += mesh_robot
    return mesh_robot_total, f3rm_frames
