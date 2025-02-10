import json
import numpy as np
from transforms3d.quaternions import mat2quat
import open3d as o3d
import glob
import os
import torch

def load_transforms(json_path):
    # Load the transformation data from `transforms_gt.json`
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["frames"]

def load_grasps_from_file(filepath):
    """Load grasp poses from a .pt file.

    Args:
        filepath (str): Path to the grasps.pt file containing palm poses [x, y, z, qx, qy, qz, qw]

    Returns:
        dict: A dictionary containing 'rot_matrix' and 'transl' for each palm pose.
    """

    print("Trying to load f3rm grasp poses")
    grasps = torch.load(filepath)._matrix
    n_samples = len(grasps)
    
    try:
        joints = torch.load(filepath.replace("grasps_to_world", "joints"))
        joint_states =  joints.detach().numpy()
    except:
        print("Joints not provided")
        joint_states = np.zeros((n_samples, 20))

    rotations = []
    translations = []

    for grasp in grasps:
        grasp = grasp.detach().numpy().T

        rotations.append(grasp[:3,:3])
        translations.append(grasp[:3,3])
    out = {
        'rot_matrix': np.array(rotations),
        'transl': np.array(translations),
        'joint_conf': joint_states,
    }
    return out

def list_directories(path):
  """
  Lists the names of all directories within a given path.

  Args:
    path: The path to search for directories.

  Returns:
    A list of strings, where each string is the name of a directory 
    found within the given path. Returns an empty list if no directories 
    are found or if the path is invalid.
  """
  if not os.path.exists(path):
    return []

  directories = []
  for entry in os.listdir(path):
    entry_path = os.path.join(path, entry)
    if os.path.isdir(entry_path):
      directories.append(entry)  # Append only the directory name
  return directories

# hithand
flange_T_grasp = np.array([
[-1.000, -0.000,  0.000, -0.000],
[0.000, -1.000,  0.000, -0.045],
[0.000,  0.000,  1.000, -0.170],
[0.000,  0.000,  0.000,  1.000],
])
flange_T_cam = np.array(
[[ 0.99975922 , 0.00395892  ,0.02158292 ,-0.0342058 ],
[ 0.00428051, -0.99988021 ,-0.01487442,  0.05933449],
[ 0.02152145,  0.01496322 ,-0.99965641, -0.03468364],
[ 0.        ,  0.         , 0.        ,  1.        ]]
)
HAND_IDX = {
    'Right_Index_0': 0,
    'Right_Index_1': 1,
    'Right_Index_2': 2,
    'Right_Index_3': 3,
    'Right_Little_0': 4,
    'Right_Little_1': 5,
    'Right_Little_2': 6,
    'Right_Little_3': 7,
    'Right_Middle_0': 8,
    'Right_Middle_1': 9,
    'Right_Middle_2': 10,
    'Right_Middle_3': 11,
    'Right_Ring_0': 12,
    'Right_Ring_1': 13,
    'Right_Ring_2': 14,
    'Right_Ring_3': 15,
    'Right_Thumb_0': 16,
    'Right_Thumb_1': 17,
    'Right_Thumb_2': 18,
    'Right_Thumb_3': 19,
}

ISAAC_IDX = ["Right_Index_0", "Right_Little_0", "Right_Middle_0", "Right_Ring_0", "Right_Thumb_0", 
            "Right_Index_1", "Right_Little_1", "Right_Middle_1", "Right_Ring_1", "Right_Thumb_1", 
            "Right_Index_2", "Right_Little_2", "Right_Middle_2", "Right_Ring_2", "Right_Thumb_2", 
            "Right_Index_3", "Right_Little_3", "Right_Middle_3", "Right_Ring_3", "Right_Thumb_3"]

ISAAC_REORDERING = []
for hand_idx in ISAAC_IDX:
    ISAAC_REORDERING.append(HAND_IDX[hand_idx])

wrist_T_grasp = np.array([
    [-0.000,  1.000, -0.000,  0.045],
    [1.000,  0.000, -0.000,  0.000],
    [0.000, -0.000, -1.000,  0.150],
    [0.000,  0.000,  0.000,  1.000],
]) # computed using ros2 run tf2_ros tf2_echo base_link_hithand f3rm_link (grasp in wrist frame) wrist_T_grasp | grasp2wrist
grasp_T_wrist = np.linalg.inv(wrist_T_grasp)

def main(
    obj_name = None,
    prompt = None,
    is_visualize = True,
    is_select_session_path = False,
    user_home = os.path.expanduser("~"),
    scene_name = "",
    model_name = ""
    ):
    scene_base_path = f"datasets/eyeinhand_nerf1/{scene_name}/"
    ycb_foundation_pose_out = scene_base_path+"foundation_pose/"
    optim_path = f"f3rm_outputs/{scene_name}/f3rm/language_pose_optimization/"
    # TODO: add isaac sim repo as submodule
    ycb_base_path = f"isaac_sim_grasping/gazebo-objects/objects_gazebo/ycb/"
    if obj_name is None:
        obj_names = [os.path.basename(d).replace(".png","") for d in glob.glob(os.path.join(scene_base_path+"masks/", "*"))]
        obj_options = {str(i): option for i, option in enumerate(obj_names)}
        obj_name = obj_options[str(input(f"{obj_options} \n Type number for object to pick: "))]
    try:
        with open(os.path.join(*ycb_foundation_pose_out.split("/"), "poses",f"{obj_name}.json"), 'r') as f:
            obj_dict = json.load(f)
    except:
        print("Cant load object:",obj_name)
        exit()

    if is_select_session_path:
        session_name = str(input("session_name: "))
    else:
        # Get all directories in the path
        dirs = [d for d in glob.glob(os.path.join(optim_path, "*")) if os.path.isdir(d)]

        # Find the newest directory based on modification time
        session_name = os.path.basename(max(dirs, key=os.path.getmtime))
    
    base_path = f"{optim_path}{session_name}/"
    dir_names = list_directories(base_path)
    options = {str(i): option for i, option in enumerate(dir_names)}
    if prompt is None:
        key = str(input(f"{options} \n Type number for {obj_name} prompt to pick: "))
        prompt = options[key]
    grasps_path = base_path+prompt+"/"

    # align_pcds(scene_base_path, is_show = True)
    grasps_data = load_grasps_from_file(grasps_path+"grasps_to_world.pt")
    frames = load_transforms(scene_base_path+"transforms_base2flange.json")
    for i, frame_data in enumerate(frames):
        rgb_path = base_path + frames[i]["file_path"]
        if obj_dict["frame"] in rgb_path:
            base_T_flange = np.array(frames[i]["transform_matrix"])
            base_T_cam = base_T_flange @ flange_T_cam
            base_T_mesh = base_T_cam @ np.array(obj_dict["cam_T_mesh"])
            mesh_T_base = np.linalg.inv(base_T_mesh)
            if is_visualize:
                obj_mesh = o3d.io.read_triangle_mesh(f"{ycb_base_path}{obj_name}/textured.obj", True)
                obj_mesh.transform(base_T_mesh)
    
    mesh_T_wrists = []
    for i,_ in enumerate(grasps_data["transl"]):
        base_T_grasp = np.eye(4)
        pre_pose = np.eye(4)

        base_T_grasp[:3,3]= grasps_data["transl"][i]
        base_T_grasp[:3,:3]= grasps_data["rot_matrix"][i]
        grasp_T_flange = np.linalg.inv(flange_T_grasp)
        base_T_flange = base_T_grasp @ grasp_T_flange

        pose_dict = {"transl": np.array([base_T_grasp[:3,3]]), "rot_matrix" : np.array([base_T_grasp[:3,:3]]), "joint_state":np.expand_dims(np.zeros((5,4)), axis=0)}
        # if is_visualize and i == 0:
        #     show_generated_grasps(scene_base_path, pose_dict, other_geometries = [obj_mesh], is_show_pcd_align=True, pcd_align_idxs = [0], is_fix=False)
        
        base_T_wrist = base_T_grasp @ grasp_T_wrist
        mesh_T_wrist = mesh_T_base @ base_T_wrist

        # Convert homogeneous matrix to quaternions
        transl = mesh_T_wrist[:3,-1]
        quat = mat2quat(mesh_T_wrist[:3,:3]) # quaternions: [w,x,y,z]
        mesh_T_wrist_xyz_q = np.concatenate((transl,quat), axis=0)
        mesh_T_wrists.append(mesh_T_wrist_xyz_q.tolist())
    grasps_data['joint_conf'][:] = grasps_data['joint_conf'][:,ISAAC_REORDERING]
    num_grasps = len(mesh_T_wrists)
    prompt = prompt.replace("-","_")

    data = {
    "gripper": "hithand", 
    "model": model_name, 
    "prompt": prompt,
    "scene_path": scene_base_path, 
    "object_id": obj_name, 
    "pose": mesh_T_wrists,
    "dofs": np.zeros((num_grasps,20)).tolist(),
    "graspit_dofs": np.zeros((num_grasps,20)).tolist(),
    "final_dofs": grasps_data["joint_conf"].tolist(),
    "fall_time": np.zeros((num_grasps)).tolist()
    }
    output_path = f"datasets/eyeinhand_nerf1/benchmark/{model_name}"
    with open(output_path+f"/hithand-{obj_name}-{model_name}-{prompt}.json", 'w') as outfile:
        # print("Created file: ", output_path)
        json.dump(data, outfile)
    print("Done")

if __name__ == "__main__":
    scene_name = "img_ycb_scene_5"
    model_name = "dex_f3rm"
    try:
        while True:
            main(scene_name = scene_name, model_name = model_name)
            
    finally:
        print("Done")
    