import json
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import copy
import os
import open3d as o3d
import torch
from jaxtyping import Float
from params_proto import ARGS
from pytorch3d.transforms import Transform3d, quaternion_to_matrix, random_quaternions
from slugify import slugify
from tqdm import tqdm

from f3rm.features.clip import clip, tokenize
from f3rm.features.clip.model import CLIP
from f3rm.features.clip_extract import CLIPArgs
from f3rm_robot.args import OptimizationArgs, validate_args
from f3rm_robot.collision import has_collision
from f3rm_robot.field_adapter import FeatureFieldAdapter, get_alpha
from f3rm_robot.initial_proposals import (
    NoProposalsError,
    dense_voxel_grid,
    marching_cubes_mask,
    otsu_mask,
    remove_statistical_outliers,
    voxel_downsample,
)
from f3rm_robot.load import LoadState, load_nerfstudio_outputs
from f3rm_robot.task import Task, get_tasks, grasp_primitives_dict
from f3rm_robot.utils import get_gripper_meshes, get_hand_meshes, get_heatmap, sample_point_cloud
from f3rm_robot.visualizer import BaseVisualizer, ViserVisualizer
from f3rm_robot.assets import get_query_frames_fk, get_query_frames_fk_torch
from f3rm_robot.benchmark_data import main as benchmark_main
args = OptimizationArgs
visualizer: Optional[BaseVisualizer] = None


def get_scene_pcd(load_state: LoadState, num_points: int, voxel_size: float) -> o3d.geometry.PointCloud:
    # Set z to -0.01, so we can show the table as well in the point cloud
    scene_min_bounds = (args.min_bounds[0], args.min_bounds[1], args.min_bounds[2])
    pcd = sample_point_cloud(load_state, num_points, scene_min_bounds, args.max_bounds)

    # Downsample and remove outliers (floaters)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    return pcd


def visualize_scene(load_state: LoadState, num_points: int = 200_000, voxel_size: float = 0.005):
    """Visualize the scene by sampling a point cloud from the NeRF and adding it to the visualizer."""
    pcd = get_scene_pcd(load_state, num_points, voxel_size)
    visualizer.add_o3d_point_cloud("scene_pcd", pcd, point_size=voxel_size + 0.001)
    return pcd


def get_qp_feats(outputs: Dict[str, torch.Tensor]) -> Float[torch.Tensor, "n c"]:
    """Get the alpha-weighted features for the given outputs from the feature field."""
    alpha = get_alpha(outputs["density"], delta=args.voxel_size)
    features = outputs["feature"]
    return alpha * features


def compute_task_embedding(task: Task) -> Float[torch.Tensor, "num_qps num_channels"]:
    """Compute the Task Embedding which is the mean of the alpha-weighted features for the given task."""
    qp_feats = get_qp_feats({"density": task.demo_density, "feature": task.demo_features})
    assert qp_feats.shape == (task.num_demos, task.num_query_points, task.num_channels)
    task_emb = qp_feats.mean(dim=0)
    return task_emb


def retrieve_task(
    query: str, clip_model: CLIP, device: torch.device, is_use_grasp_prompt: bool = True
) -> Tuple[Task, Float[torch.Tensor, "num_qps num_channels"], Float[torch.Tensor, "1 num_channels"]]:
    """
    Retrieve the most relevant task for a given query. Returns the Task, task embedding, and query embedding.
    """
    # Retrieve relevant demonstrations by encoding query using CLIP and comparing it to the task embeddings.
    with torch.no_grad():
        tokens = tokenize(query).to(device)
        query_emb = clip_model.encode_text(tokens)
        query_emb /= query_emb.norm(dim=-1, keepdim=True)
    if is_use_grasp_prompt:
        with torch.no_grad():
            grasp_descriptions = list(grasp_primitives_dict.keys())
            primitive_sims = {}
            for i, grasp_description in enumerate(grasp_descriptions):
                grasp_tokens = tokenize(grasp_description).to(device)
                grasp_query_emb = clip_model.encode_text(grasp_tokens)
                grasp_query_emb /= grasp_query_emb.norm(dim=-1, keepdim=True)
                prim_query_sim = torch.cosine_similarity(query_emb, grasp_query_emb)
                primitive_sims[grasp_description] = prim_query_sim
        most_sim_grasp = max(primitive_sims, key=primitive_sims.get)        
        print(f"Found most similar grasp type {most_sim_grasp}: {grasp_primitives_dict[most_sim_grasp]}")
        tasks = get_tasks(grasp_primitives_dict[most_sim_grasp])
    else:
        # Compute mean embedding for each task, and compare to the query
        tasks = get_tasks()
    task_embs = torch.stack([compute_task_embedding(t) for t in tasks]).to(device)
    mean_task_embs = task_embs.mean(dim=1)
    task_sims = torch.cosine_similarity(query_emb, mean_task_embs)

    # Select task with the highest similarity to the query
    task_idx = torch.argmax(task_sims)
    task_emb = task_embs[task_idx]
    return tasks[task_idx], task_emb, query_emb


def get_initial_voxel_grid(
    feature_field: FeatureFieldAdapter, query: str, clip_model: CLIP, device: torch.device
) -> Tuple[Float[torch.Tensor, "num_voxels 3"], Float[torch.Tensor, "num_voxels"], Dict[str, int]]:
    """
    Get the initial masked voxel grid based on density (alpha) and language (CLIP features). These correspond to the
    coarse (x, y, z) proposals.

    Returns voxel grid as a tensor of shape (num_voxels, 3), voxel similarities with language, and a dict with mettrics.
    """
    # Firstly, we sample a dense voxel grid over the workspace and use marching cubes to only get the surface.
    voxel_size = args.voxel_size
    voxel_grid = dense_voxel_grid(args.min_bounds, args.max_bounds, voxel_size).to(device)
    og_voxel_grid_shape = voxel_grid.shape
    voxel_grid = voxel_grid.reshape(-1, 3)
    metrics = {"initial": len(voxel_grid)}

    # Initial alpha masking (i.e., density). Use marching cubes to only get surface.
    with torch.no_grad():
        alpha = feature_field.get_alpha(voxel_grid, voxel_size)
    alpha_vg = alpha.reshape(og_voxel_grid_shape[:-1])
    voxel_grid = marching_cubes_mask(alpha_vg, args.alpha_threshold, args.min_bounds, args.max_bounds)
    metrics["mcubes_masked"] = len(voxel_grid)

    # Down sample and remove outliers to get rid of floaters.
    voxel_grid = voxel_downsample(voxel_grid, voxel_size)
    voxel_grid, _ = remove_statistical_outliers(voxel_grid, num_points=50, std_ratio=4.0)
    metrics["downsampled_remove_outliers"] = len(voxel_grid)

    print(f"Number of voxels after masking using NeRF density: {len(voxel_grid)}")
    if args.visualize:
        with torch.no_grad():
            rgb = feature_field.get_rgb(voxel_grid)
        visualizer.add_point_cloud(
            "initial_proposals/alpha_masked",
            voxel_grid.cpu().numpy(),
            rgb.cpu().numpy(),
            point_size=voxel_size,
            visible=False,
        )

    # Feature masking by comparing each voxel's feature with the user query and negatives
    queries = [query, "object", "things", "stuff", "texture", "robot hand", "robot arm"]  # we use the negatives from LERF
    with torch.no_grad():
        tokens = tokenize(queries).to(device)
        query_embs = clip_model.encode_text(tokens).float()
        query_embs /= query_embs.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        outputs = feature_field(voxel_grid)
    voxel_feats = get_qp_feats(outputs)
    voxel_feats /= voxel_feats.norm(dim=-1, keepdim=True)

    # Compute softmax over similarities between voxel features and query embeddings
    voxel_sims = voxel_feats @ query_embs.T
    probs = voxel_sims / args.softmax_temperature
    probs = probs.softmax(dim=-1)
    probs = torch.nan_to_num_(probs, nan=1e-7)

    # Sample from the distribution, 0-index is the positive query
    labels = torch.multinomial(probs, num_samples=1)
    softmax_mask = (labels == 0).squeeze()
    voxel_grid = voxel_grid[softmax_mask]
    voxel_sims = voxel_sims[:, 0][softmax_mask]

    # If no voxel sims, then the query didn't match to anything so raise error
    if len(voxel_grid) == 0:
        raise NoProposalsError(
            f'No proposals found for query "{query}" after language masking. Try use a different query.'
        )
    
    # filter and sort voxels accoding to similarity
    k = min(args.max_voxels, len(voxel_grid))
    _, top_indices = torch.topk(voxel_sims, k)
    voxel_grid = voxel_grid[top_indices]
    voxel_sims = voxel_sims[top_indices]
    
    # output
    metrics["language_masked"] = len(voxel_grid)
    print(f"Number of voxels after language masking using CLIP features: {len(voxel_grid)}")
    if args.visualize:
        visualizer.add_point_cloud(
            "initial_proposals/lang_probs",
            voxel_grid.cpu().numpy(),
            get_heatmap(voxel_sims),
            point_size=voxel_size,
            visible=False,
        )
    return voxel_grid, voxel_sims, metrics


def get_language_guidance_fn(voxel_sims: Float[torch.Tensor, "num_voxels"], query_emb: Float[torch.Tensor, "1 c"]):
    """
    Get the function for computing the language guidance given query point features and the embedded user query.
    This works well in our experiments, but you may need to tune it for your environment and use case.
    """
    lang_loss_fn = torch.nn.CosineSimilarity()
    feat_mask, _ = otsu_mask(voxel_sims)
    remaining_voxel_sims = voxel_sims[feat_mask]
    sim_min = remaining_voxel_sims.min()
    sim_max = remaining_voxel_sims.max()

    def language_guidance(qp_feats):
        qp_mean_feats = qp_feats.mean(dim=1)
        lang_losses = lang_loss_fn(qp_mean_feats, query_emb)
        lang_losses = (lang_losses - sim_min) / (sim_max - sim_min)

        # We consider the guidance as a multiplier. Since pose loss is negative cosine similarity, we want the
        # multiplier to be higher when the proposal matches the language query.
        lang_multiplier = 1 + lang_losses
        # Don't let multiplier go below 0 as positive pose loss with negative multiplier can mess things up
        lang_multiplier = torch.clamp(lang_multiplier, min=0)
        return lang_multiplier

    return language_guidance

def get_preshape(task, repeat, device, num_links, num_fingers):
    task_joints = task.demo_joints.to(device)[0]
    if num_links * num_fingers == 15:
        task_joints = task_joints.reshape(5,4)[:,:3].flatten()
    return task_joints.repeat(repeat, 1)

def get_complete_joints(base_joints, top_joints, num_links, num_fingers):
    base_joints_reshape = base_joints.reshape(-1,num_fingers,1)
    top_joints_reshape = top_joints.reshape(-1,num_fingers,num_links-1)
    if num_links * num_fingers == 15:
        last_joints = top_joints_reshape[:,:,-1].unsqueeze(dim=2)
        complete_joints = torch.cat([base_joints_reshape, top_joints_reshape, last_joints],dim=2)
    else:
        complete_joints = torch.cat([base_joints_reshape, top_joints_reshape],dim=2)
    return complete_joints.reshape(-1,4*num_fingers)

def language_pose_optimization(
    feature_field: FeatureFieldAdapter, 
    clip_model: CLIP, 
    query: str, 
    device: torch.device,
    num_links: int = 4,
    num_fingers: int = 5,
    is_qp_fk: bool = True,
    is_init_coll_joint_check : bool = False,
    is_zero_init: bool = False,
    is_show_hand_opt: bool = True,
    is_save_gripper_meshes: bool = False,
    is_optim_less_joints: bool = True,
    is_use_grasp_prompt: bool = True,
    is_output_less: bool = False,

) -> Dict[str, Any]:
    """
    Optimize 6-DOF poses for the given language query. We return the ranked grasps after optimization and the metrics.
    """
    metrics = {"query": query}

    # Retrieve the relevant task for the query, and compute the task embedding
    task, task_emb, query_emb = retrieve_task(query, clip_model, device, is_use_grasp_prompt)
    task_emb = task_emb.reshape(-1)  # [num_qps * num_channels]
    query_points = task.query_points.to(device)

    if hasattr(task, 'link_points'):
        is_og_f3rm = task.link_points is None
    else:
        is_og_f3rm = True

    if is_og_f3rm:
        is_qp_fk = False
        is_zero_init = True
        is_init_coll_joint_check = False
        args.num_steps = 200
    else:
        link_points = task.link_points.to(device)
    print(f'Matched "{query}" to task {task.name}.')
    metrics["retrieved_task"] = task.name

    # Get coarse voxel grid proposals using alpha and language-masking.
    voxel_grid, voxel_sims, metrics["num_voxels"] = get_initial_voxel_grid(feature_field, query, clip_model, device)

    # Sample rotations for each voxel to get the initial 6-DOF proposals. We parametrize rotations as quaternions and
    # multiply by a scale factor so the scale is more reasonable for optimization. You can tune this to your liking.
    translations = voxel_grid.repeat_interleave(args.num_rots_per_voxel, dim=0)
    rotations = random_quaternions(len(translations), device=device)

    # use selected demo joint values / preshape to initialize.
    if is_og_f3rm:
        joint_demo = torch.zeros(len(translations), num_links*num_fingers, device = device)
    else:
        joint_demo = get_preshape(task, len(translations), device, num_links, num_fingers)
    
    if is_zero_init:
        joints_pre = torch.zeros(len(translations), num_links*num_fingers, device = device)
    else:
        min_rad = -0.2
        max_rad = 0.1
        joints_init = torch.FloatTensor(len(translations), num_links*num_fingers).uniform_(min_rad, max_rad).to(device = device)
        joints_pre = joint_demo + joints_init
        # joints_pre = torch.rand(len(translations), num_links*num_fingers, device = device)*0.8

    joints_pre[:, ::num_links] = joint_demo[:, ::num_links]
    # joints_pre[:, ::num_links] = 0
    
    if is_optim_less_joints:
        joints = joints_pre.reshape(-1, num_fingers, num_links)[:,:,1:].reshape(-1, (num_links*num_fingers) - num_fingers)
        base_joints = joints_pre.reshape(-1, num_fingers, num_links)[:,:,0]
    else:
        joints = joints_pre
    joints = torch.clamp(joints, 0.0, 1.57)
    rot_scale = 0.1
    rotations = rotations * rot_scale
    metrics["num_proposals"] = {"initial": len(translations)}

    def get_rotation_mats(rotations_):
        """Convert quaternions back into rotation matrices."""
        # Normalize the quaternions so they're unit and valid rotations
        rotations_ = rotations_ / rotations_.norm(dim=-1, keepdim=True)
        return quaternion_to_matrix(rotations_ * (1 / rot_scale))

    def get_grasps_to_world(translations_, rotations_):
        """Convert translations and rotations into Transform3d."""
        rotation_mats_ = get_rotation_mats(rotations_)
        # We need to transpose because Transform3d uses row vector rather than column vector convention
        return Transform3d(device=device).rotate(rotation_mats_.transpose(1, 2)).translate(translations_)

    # Remove initial grasps in collision. We did not optimize our collision checking, so it is a bit slow.
    grasps_to_world = get_grasps_to_world(translations, rotations)
    print(f'Checking initial collisions for {grasps_to_world.get_matrix().size()}')
    with torch.no_grad():
        if is_init_coll_joint_check:
            if is_optim_less_joints:
                joints_complete = get_complete_joints(base_joints, joints, num_links, num_fingers)
                collision_detected = has_collision(feature_field, grasps_to_world, joints_complete)
            else:
                collision_detected = has_collision(feature_field, grasps_to_world, joints)
        else:
            collision_detected = has_collision(feature_field, grasps_to_world, None)
            
    translations = translations[~collision_detected]
    rotations = rotations[~collision_detected]
    joints = joints[~collision_detected]
    if is_optim_less_joints:
        base_joints = base_joints[~collision_detected]

    metrics["num_proposals"]["initial_cfree"] = len(translations)
    print(f"Number of 6-DOF proposals: {len(translations)}.")
    print('Done checking initial collisions')

    # Shuffle the remaining proposals
    permutation = torch.randperm(len(translations), device=device)
    translations = translations[permutation]
    rotations = rotations[permutation]
    joints = joints[permutation]
    if is_optim_less_joints:
        base_joints = base_joints[permutation]
    # Setup optimizer
    translations.requires_grad_()
    rotations.requires_grad_()
    joints.requires_grad_()
    optimizer = torch.optim.Adam([translations, rotations, joints], lr=args.lr)
    pose_loss_fn = torch.nn.CosineSimilarity()
    language_guidance_fn = get_language_guidance_fn(voxel_sims, query_emb)
    batch_size = args.ray_samples_per_batch // len(query_points)
    step_losses = None

    # Now we can optimize!
    for step in tqdm(range(args.num_steps), desc=f'Optimizing poses for "{query}"'):
        optimizer.zero_grad()
        all_grasps_to_world = []
        all_joints = []
        step_losses = []
        num_proposals = len(translations)

        for i in range(0, num_proposals, batch_size):
            if is_optim_less_joints:
                batch_base_joints = base_joints[i : i + batch_size]
            else:
                with torch.no_grad():
                    joint_demo = get_preshape(task, len(joints), device)
                    joints[:, ::num_links] = joint_demo[:, ::num_links]
            batch_translations = translations[i : i + batch_size]
            batch_rotations = rotations[i : i + batch_size]
            batch_joints = joints[i : i + batch_size]

            # Transform query points by the proposals, and forward through the feature field
            grasps_to_world = get_grasps_to_world(batch_translations, batch_rotations)

            all_grasps_to_world.append(grasps_to_world)

            if is_optim_less_joints:
                batch_joints_complete = get_complete_joints(batch_base_joints, batch_joints, num_links, num_fingers)
                all_joints.append(batch_joints_complete)
            else:
                all_joints.append(batch_joints)

            if is_qp_fk and is_optim_less_joints:
                batch_joints_complete = get_complete_joints(batch_base_joints, batch_joints, num_links, num_fingers)
                f3rm_fk_tf = get_query_frames_fk_torch(batch_joints_complete.reshape(-1,num_fingers,4))
            elif is_qp_fk:
                f3rm_fk_tf = get_query_frames_fk_torch(batch_joints.reshape(-1,num_fingers,4))
            if is_qp_fk:
                b, j, c, r = f3rm_fk_tf.shape # batch_size, n_joints, tf_column, tf_row
                f3rm_frames = Transform3d(matrix=f3rm_fk_tf.reshape(b * j, c, r))
                query_points = f3rm_frames.transform_points(link_points).reshape(b, j * link_points.shape[0], link_points.shape[1])

            qps = grasps_to_world.transform_points(query_points)
            outputs = feature_field(qps)
            qp_feats = get_qp_feats(outputs)

            # Compute pose loss and language guidance
            pose_loss = -pose_loss_fn(qp_feats.flatten(1, 2), task_emb)
            lang_guidance = language_guidance_fn(qp_feats)
            batch_losses = lang_guidance * pose_loss
            loss = batch_losses.mean()
            loss.backward()
            step_losses.append(batch_losses.detach())

        # Optimizer step
        optimizer.step()
        with torch.no_grad():
            joints.copy_(torch.clamp(joints, 0, 1.57))
        step_losses = torch.cat(step_losses)

        # Visualize top poses. Note this does not take into account collisions.
        if args.visualize:
            sorted_losses, sorted_indices = step_losses.sort(descending=False)
            best_losses, best_indices = (
                sorted_losses[: args.num_poses_to_visualize],
                sorted_indices[: args.num_poses_to_visualize],
            )
            all_grasps_to_world = Transform3d.stack(*all_grasps_to_world)
            best_grasps_to_world = all_grasps_to_world[best_indices]
            best_joints = torch.cat(all_joints).squeeze()[best_indices]
            # We use jet cmap as viser lighting is a bit messed up for turbo
            heatmap = torch.from_numpy(get_heatmap(best_losses, invert=True, cmap_name="jet")).to(device)
            if is_show_hand_opt:
                all_verts, all_faces = get_hand_meshes(best_grasps_to_world, best_joints)
                # all_verts, all_faces = get_hand_meshes(best_grasps_to_world, None)
                for idx, (verts, faces) in enumerate(zip(all_verts, all_faces)):
                    visualizer.add_mesh(f"grasps/grasp_{idx + 1}", verts, faces, heatmap[idx])
            else:
                for idx, (verts, faces) in enumerate(zip(*get_gripper_meshes(best_grasps_to_world))):
                    visualizer.add_mesh(f"grasps/grasp_{idx + 1}", verts, faces, heatmap[idx])

        # Prune proposals
        if args.keep_proportion < 1.0 and num_proposals > args.min_proposals and step > args.prune_after:
            new_num_proposals = max(int(args.keep_proportion * num_proposals), args.min_proposals)
            losses, best_indices = torch.topk(step_losses, k=new_num_proposals, largest=False)
            translations = translations[best_indices].detach().clone()
            rotations = rotations[best_indices].detach().clone()
            joints = joints[best_indices].detach().clone()
            if is_optim_less_joints:
                base_joints = base_joints[best_indices].detach().clone()
            # Need to set up optimizer again
            translations.requires_grad_()
            rotations.requires_grad_()
            joints.requires_grad_()
            optimizer = torch.optim.Adam([translations, rotations, joints], lr=args.lr)
            metrics["num_proposals"][f"pruned_step_{step:04d}"] = new_num_proposals

    # Optimization finished, check remaining grasps for collisions
    grasps_to_world = get_grasps_to_world(translations, rotations)
    print(f'Checking final collisions for {grasps_to_world.get_matrix().size()}')
    with torch.no_grad():
        if is_optim_less_joints:
            joints = get_complete_joints(base_joints, joints, num_links, num_fingers)
            collision_detected = has_collision(feature_field, grasps_to_world, joints, is_final=True)
        else:
            collision_detected = has_collision(feature_field, grasps_to_world, joints, is_final=True)
        # collision_detected = has_collision(feature_field, grasps_to_world, None, is_final=True)
    print(f"Removed {collision_detected.sum()} of {len(grasps_to_world)} optimized proposals in collision")
    grasps_to_world = grasps_to_world[~collision_detected]
    joints = joints[~collision_detected]
    print(f'Final number of 6-DOF proposals for "{query}": {len(grasps_to_world)}')
    metrics["num_proposals"]["final_cfree"] = len(grasps_to_world)
    print('Done checking final collisions.')
    # Sort the grasps by their losses before returning
    masked_losses = step_losses[~collision_detected]
    sorted_losses, sorted_indices = masked_losses.sort(descending=False)
    grasps_to_world = grasps_to_world[sorted_indices]
    joints = joints[sorted_indices]
    if is_output_less:
        num_outs = args.num_outs
        num_outs = min(num_outs, len(joints))
        joints = joints[:num_outs]
        grasps_to_world = grasps_to_world[:num_outs]
    results = {"grasps_to_world": grasps_to_world, "joints": joints ,"metrics": metrics}
    # Show the best grasps without collisions
    if args.visualize:
        best_losses = sorted_losses[: args.num_poses_to_visualize]
        best_grasps_to_world = grasps_to_world[: args.num_poses_to_visualize]
        joints = joints[: args.num_poses_to_visualize]
        # all_verts, all_faces = get_hand_meshes(best_grasps_to_world, None)
        all_verts, all_faces = get_hand_meshes(best_grasps_to_world, joints)
        # We use jet cmap as viser lighting is a bit messed up for turbo
        heatmap = torch.from_numpy(get_heatmap(best_losses, invert=True, cmap_name="jet")).to(device)
        gripper_meshes = []
        for idx, (verts, faces) in enumerate(zip(all_verts, all_faces)):
            visualizer.add_mesh(f"grasps/grasp_{idx + 1}", verts, faces, heatmap[idx])
            if is_save_gripper_meshes:
                gripper_mesh = o3d.geometry.TriangleMesh()
                gripper_mesh.vertices = o3d.utility.Vector3dVector(verts)
                gripper_mesh.triangles = o3d.utility.Vector3iVector(faces)
                gripper_mesh.paint_uniform_color(heatmap[idx].cpu().numpy())
                gripper_meshes.append(gripper_mesh)
        # Single mesh with all the grippers
        if is_save_gripper_meshes:
            gripper_mesh = reduce(lambda a, b: a + b, gripper_meshes)
            results["gripper_mesh"] = gripper_mesh

    return results


def entrypoint():
    ARGS.parse_args()
    print(ARGS.groups.keys())
    validate_args()
    is_benchmark = any(name in args.scene for name in args.benchmarks)
    # Load feature field
    print(f"Loading feature field from {args.scene}...")
    load_state = load_nerfstudio_outputs(args.scene)
    device = load_state.pipeline.device
    feature_field = load_state.feature_field_adapter()

    # Setup output directory and save args
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.scene).parent / "language_pose_optimization" / time_stamp
    output_dir.mkdir(parents=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Visualize scene
    global visualizer
    if args.visualize:
        visualizer = ViserVisualizer(args.viser_host, args.viser_port)
        scene_pcd = visualize_scene(load_state)
        o3d.io.write_point_cloud(str(output_dir / "scene.ply"), scene_pcd)
        print(f"Saved scene point cloud to {output_dir / 'scene.ply'}")

    # Load CLIP so we can query via language
    print("Loading CLIP model...", end=" ")
    clip_model, _ = clip.load(CLIPArgs.model_name, device=device)
    print("Done!")

    # Ask for query from user and optimize. If we're using the ViserVisualizer, we can use a textbox in the GUI
    if isinstance(visualizer, ViserVisualizer):
        input_fn, enable_gui = visualizer.add_query_gui()
        print(f"Enter query in the visualizer at: {visualizer.url}")
    else:
        input_fn = lambda: input("Enter query (empty to exit): ")
        enable_gui = lambda: None

    queries = []
    benchmark_data = {"prompts":["",]}
    if is_benchmark:
        # Input path
        input_path = Path(args.scene)

        # Construct the new path
        benchmark_data_path = Path(args.benchmark_path) / input_path.parts[1] / "scene_benchmark_data.json"

        with open(benchmark_data_path) as f:
            benchmark_data = json.load(f)
            print(benchmark_data)
        scene_name = benchmark_data["scene_path"].split("/")[1]

    while True and benchmark_data["prompts"]:
        if not is_benchmark:
            enable_gui()
        try:
            if is_benchmark:
                query = benchmark_data["prompts"].pop().replace("_"," ")
                obj_name = benchmark_data["object_ids"].pop()
                
            else:
                query = input_fn().strip()
        except KeyboardInterrupt:
            print()
            break
        if query == "":
            break

        # Optimize for the query!
        try:
            results = language_pose_optimization(feature_field, clip_model, query, device, is_show_hand_opt=not is_benchmark, is_use_grasp_prompt=args.is_use_grasp_prompt, is_output_less = is_benchmark)
        except NoProposalsError as e:
            # Print error message
            print(e)
            continue
        queries.append(query)

        # Write results to directory
        query_dir = output_dir / slugify(query)
        query_dir.mkdir(parents=True, exist_ok=True)
        with open(query_dir / "metrics.json", "w") as f:
            json.dump(results["metrics"], f, indent=4)
        torch.save(results["grasps_to_world"].cpu(), query_dir / "grasps_to_world.pt")
        torch.save(results["joints"].cpu(), query_dir / "joints.pt")
        if "gripper_mesh" in results:
            o3d.io.write_triangle_mesh(str(query_dir / "gripper_mesh.ply"), results["gripper_mesh"])
        print(f"Saved results to {query_dir}")

        # Write queries to file. Save inside loop so we get partial results if we crash
        with open(output_dir / "queries.json", "w") as f:
            json.dump(queries, f, indent=4)
        
        if is_benchmark:
            benchmark_main(
                obj_name = obj_name, 
                prompt = query.replace(" ","-"), 
                scene_name = scene_name, 
                model_name = args.model_name
            )
    print(f"Results saved to {output_dir}")
    print("Exiting...")


if __name__ == "__main__":
    entrypoint()
