# import math
# import pytorch_kinematics as pk
# import numpy as np
# # load robot description from URDF and specify end effector link
# # chain = pk.build_serial_chain_from_urdf(open("f3rm_robot/assets/hithand_palm/hithand.urdf", mode="rb").read())
# chain = pk.build_chain_from_urdf(open("f3rm_robot/assets/hithand_palm/hithand.urdf", mode="rb").read())
# # prints out the (nested) tree of links
# print(chain)
# # prints out list of joint names
# print(chain.get_joint_parameter_names())

# # specify joint values (can do so in many forms)
# th = np.zeros(20)
# # do forward kinematics and get transform objects; end_only=False gives a dictionary of transforms for all links
# fk_link = chain.forward_kinematics(th)
# print(fk_link)
# # look up the transform for a specific link
# tg = fk_link['f3rm_little_dist']
# # get transform matrix (1,4,4), then convert to separate position and unit quaternion
# m = tg.get_matrix()
# print(m)
# pos = m[:, :3, 3]
# rot = pk.matrix_to_quaternion(m[:, :3, :3])
import numpy as np
import torch
import pytorch_kinematics as pk
from urdfpy import URDF

# Dummy functions for asset path (replace with your actual implementation)
def get_asset_path(relative_path):
  """
  This is a placeholder function. Replace it with your actual logic to retrieve asset paths.
  """
  return f"f3rm_robot/assets/{relative_path}"  # Assuming assets are in f3rm_robot/assets

# --- Transformation matrices ---
wrist_T_grasp = np.array([
    [-0.000, 1.000, -0.000, 0.045],
    [1.000, 0.000, -0.000, 0.000],
    [0.000, -0.000, -1.000, 0.150],
    [0.000, 0.000, 0.000, 1.000],
])

grasp_T_wrist = np.linalg.inv(wrist_T_grasp)

# Convert to PyTorch tensors
wrist_T_grasp_tensor = torch.from_numpy(wrist_T_grasp).float()
grasp_T_wrist_tensor = torch.from_numpy(grasp_T_wrist).float()

# --- Load URDF ---
asset_path = get_asset_path("hithand_palm/hithand.urdf")
asset_path_coll = get_asset_path("hithand_palm/hithand_collision.urdf")

# using urdf_parser_py
robot_og = URDF.load(asset_path)
robot_coll = URDF.load(asset_path_coll)

# Using pytorch_kinematics (loading both for collision and visualization)
chain_og = pk.build_chain_from_urdf(open("f3rm_robot/assets/hithand_palm/hithand.urdf", mode="rb").read())
chain_coll = pk.build_chain_from_urdf(open("f3rm_robot/assets/hithand_palm/hithand_collision.urdf", mode="rb").read())


# --- Extract finger root link names ---
f3rm_names = {link.name for link in robot_og.links if "f3rm" in link.name}

# Create a mapping from link name to index for efficient lookup later. 
# Consider only links present in the collision model.
f3rm_names_to_idx = {link.name: i for i, link in enumerate(robot_coll.links) if link.name in f3rm_names}


# --- Helper functions ---
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
    if joint.ndim > 1:
        transform_list = []
        for j in joint:
            fk_link = chain_coll.forward_kinematics(j)
            transform_list.append(torch.stack([grasp_T_wrist_tensor @ fk_link[name].get_matrix().squeeze() for name in f3rm_names_to_idx.keys()]))
        f3rm_frames = torch.stack(transform_list)

    else:
      fk_link = chain_coll.forward_kinematics(joint)

      if is_debug:
          print(f"names: {f3rm_names_to_idx}")
          print(fk_link)

      if is_return_tensor:
          f3rm_frames = torch.stack([grasp_T_wrist_tensor @ fk_link[name].get_matrix().squeeze() for name in f3rm_names_to_idx.keys()])
      else:
          f3rm_frames = {f"grasp_T_{name}": grasp_T_wrist_tensor @ fk_link[name].get_matrix().squeeze() for name in f3rm_names_to_idx.keys()}

    return f3rm_frames

# --- Example Usage ---
# Assuming you have joint angles as a NumPy array
joint_angles_np = np.random.rand(5, 4)  # Example: 5 fingers, 4 joints each

# Convert to PyTorch tensor
joint_angles = torch.from_numpy(joint_angles_np).float()

# Get forward kinematics
f3rm_frames_tensor = get_query_frames_fk_torch(joint_angles, is_return_tensor=True)

# Print the shape of the resulting tensor
print("f3rm_frames_tensor.shape:", f3rm_frames_tensor.shape)

# Or get a dictionary of transforms
f3rm_frames_dict = get_query_frames_fk_torch(joint_angles, is_return_tensor=False)

# Print the keys of the dictionary
print("f3rm_frames_dict.keys():", f3rm_frames_dict.keys())

# Example Usage with batched joint angles:
joint_angles_np_batched = np.random.rand(10, 5, 4) # (batch_size, 5, 4)
joint_angles_batched = torch.from_numpy(joint_angles_np_batched).float()
f3rm_frames_tensor_batched = get_query_frames_fk_torch(joint_angles_batched, is_return_tensor=True)
print("f3rm_frames_tensor_batched.shape:", f3rm_frames_tensor_batched.shape)