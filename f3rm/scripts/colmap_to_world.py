import json
from pathlib import Path

import numpy as np
import open3d as o3d


def _compute_translation_3D(A, B, R):
    """Compute the translation between A and B given the rotation matrix R"""
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    t = -R @ centroid_A + centroid_B

    return t


from typing import List, Dict, Tuple
import math


def _isRotm(R: np.ndarray) -> bool:
    """Checks if a matrix is a valid rotation matrix."""
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def _rotm_to_eulerXYZ(R: np.ndarray) -> np.ndarray:
    """Calculates rotation matrix to euler angles"""
    assert _isRotm(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def get_transformation_matrix(
    colmap_frames: List[Dict],
    real_frames: List[Dict],
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
    trans_error_threshold: float = 0.005,
    rot_error_threshold: float = 1.5,
) -> Tuple[np.ndarray, dict]:
    """
    Print errors between reprojected COLMAP points in real world coordinates,
    and the actual recorded real robot poses.
    """
    pos_errors = []
    rot_errors = []

    scale_mat = np.array(
        [[scale, 0, 0, 0], [0, scale, 0, 0], [0, 0, scale, 0], [0, 0, 0, 1]]
    )
    rot_mat = np.eye(4)
    rot_mat[:3, :3] = rotation

    transform_mat = rot_mat @ scale_mat
    transform_mat[:3, -1] = translation
    print(transform_mat)

    print("======================")
    for idx, (colmap_frame, real_frame) in enumerate(zip(colmap_frames, real_frames)):
        colmap_transform = np.array(colmap_frame["transform_matrix"])
        pred_rotm = rotation @ colmap_transform[:3, :3]
        pred_eulerXYZ = _rotm_to_eulerXYZ(pred_rotm)

        real_transform = np.array(real_frame["transform_matrix"])
        real_rotm = real_transform[:3, :3]
        real_eulerXYZ = _rotm_to_eulerXYZ(real_rotm)
        diff_eulerXYZ = np.rad2deg(pred_eulerXYZ - real_eulerXYZ)
        rot_errors.append(diff_eulerXYZ)

        pred_pos = transform_mat @ np.append(colmap_transform[:3, -1], 1)
        pred_pos = pred_pos[:3]
        pred_pos_check = (rotation @ colmap_transform[:3, -1]) * scale + translation
        assert np.allclose(pred_pos, pred_pos_check), "things are broken!"

        real_pos = real_transform[:3, -1]
        diff_xyz = pred_pos - real_pos
        pos_errors.append(diff_xyz)

    for name, errors in (
        ("Position (in m)", pos_errors),
        ("Rotation (in deg)", rot_errors),
    ):
        errors = np.abs(np.array(errors))
        print(f"=== {name} ===")
        print("Avg error:", errors.mean(0))
        print("Max error:", errors.max(0))
        print("Min error:", errors.min(0))

    mean_trans_error = np.mean(np.abs(np.array(pos_errors)))
    if mean_trans_error > trans_error_threshold:
        raise ValueError(f"Position error {mean_trans_error} is too high")
    mean_rot_error = np.mean(np.abs(np.array(rot_errors)))
    if mean_rot_error > rot_error_threshold:
        raise ValueError(f"Rotation error {mean_rot_error} is too high")

    errors = {
        "position_avg": mean_trans_error,
        "rot_deg_avg": mean_rot_error,
    }
    return transform_mat, errors


def run(scan_dir):
    scan_dir = Path(scan_dir)
    if (scan_dir / "colmap_to_world.json").exists():
        print(f"colmap_to_world.json already exists in {scan_dir}")
        return

    with open(scan_dir / "transforms.json") as f:
        transforms = json.load(f)

    with open(scan_dir / "transforms_gt.json") as f:
        transforms_gt = json.load(f)

    gt_frames = list(sorted(transforms_gt["frames"], key=lambda f: f["file_path"]))
    colmap_frames = list(sorted(transforms["frames"], key=lambda f: f["file_path"]))

    scales = []
    colmap_homeJ = np.array(colmap_frames[0]["transform_matrix"])
    gt_homeJ = np.array(gt_frames[0]["transform_matrix"])
    rotation = gt_homeJ[:3, :3] @ np.linalg.inv(colmap_homeJ)[:3, :3]
    print(f"Rotation:\n{rotation}")

    for colmap_frame, gt_frame in zip(colmap_frames[1:], gt_frames[1:]):
        colmap_pos = np.array(colmap_frame["transform_matrix"])[:3, -1]
        gt_pos = np.array(gt_frame["transform_matrix"])[:3, -1]

        colmap_dist = np.linalg.norm(colmap_homeJ[:3, -1] - colmap_pos)
        gt_dist = np.linalg.norm(gt_homeJ[:3, -1] - gt_pos)
        scales.append(colmap_dist / gt_dist)

    scale = np.mean(scales)
    scale = 1 / scale
    print(f"Scale: {scale} +- {np.std(scales)}")

    colmap_pts = []
    real_pts = []
    for colmap_frame, real_frame in zip(colmap_frames, gt_frames):
        #     print(colmap_frame['file_path'], real_frame['file_path'])
        # assert colmap_frame["file_path"] == real_frame["file_path"]
        colmap_transform = np.array(colmap_frame["transform_matrix"])
        real_transform = np.array(real_frame["transform_matrix"])
        colmap_pts.append(colmap_transform[:3, -1])
        real_pts.append(real_transform[:3, -1])

    colmap_pts = np.array(colmap_pts) * scale
    real_pts = np.array(real_pts)
    translation = _compute_translation_3D(colmap_pts.T, real_pts.T, rotation).reshape(
        (3,)
    )
    print(f"Translation: {translation}")

    transformation_matrix, errors = get_transformation_matrix(
        colmap_frames, gt_frames, rotation, scale, translation
    )
    print("=== FINAL TRANSFORMATION MATRIX ===")
    print(transformation_matrix)

    colmap_to_world = {
        "transformation_matrix": transformation_matrix.tolist(),
        "errors": errors,
        "scan_dir": str(scan_dir),
    }
    with open(scan_dir / "colmap_to_world.json", "w") as f:
        json.dump(colmap_to_world, f, indent=2)
    print(f"Wrote colmap_to_world.json to {scan_dir}")


if __name__ == "__main__":
    with open("calibration_dir.txt", "r") as f:
        calibration_dir = f.read().strip()
    print(f"calibration_dir: {calibration_dir}")
    run(calibration_dir)