#%%
import re
import json
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as rot
from pathlib import Path
from typing import Dict, List, Tuple, Optional
# -------------------------- Image matching helpers -------------------------- #

def brute_force_match_indices_by_mse(
    colmap_paths: List[Path] | List[str],
    orig_paths: List[Path] | List[str],
    size: Tuple[int, int] | None = (320, 240),
    ignore_white: bool = False,
    white_thresh: int = 250,
    ignore_transparent: bool = True,
    chunk_size: int = 16,
    return_scores: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Brute-force match each COLMAP image to the most similar original image by MSE.

    Args:
      colmap_paths: list of image paths for COLMAP-rendered (or exported) images.
      orig_paths:   list of image paths for the original image stream (saved in order).
      size:         (W,H) to resize images for matching. None = no resize.
      ignore_white: if True, ignore near-white pixels (useful for white backgrounds).
      white_thresh: RGB>white_thresh considered white (0-255 scale).
      ignore_transparent: if True, ignore pixels with alpha==0 when present.
      chunk_size:   compute in chunks to keep memory bounded.
      return_scores: if True, also return the MSE score matrix (N_colmap x N_orig)

    Returns:
      indices: np.int64 array of shape (N_colmap,), where indices[i] = j gives
               the index in orig_paths best matching colmap_paths[i].
      scores:  optional float32 array (N_colmap, N_orig) of MSE values.
    """
    import cv2

    def load_and_preprocess(paths: List[Path] | List[str]):
        imgs: List[np.ndarray] = []
        masks: List[np.ndarray] = []  # boolean masks of valid pixels
        for p in paths:
            p = str(p)
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise FileNotFoundError(f"Failed to read image: {p}")
            # Split alpha if present
            alpha = None
            if im.ndim == 3 and im.shape[2] == 4:
                alpha = im[:, :, 3]
                im = im[:, :, :3]
            # Convert to gray for MSE
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if size is not None:
                W, H = size
                gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
                if alpha is not None:
                    alpha = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_NEAREST)
            gray = gray.astype(np.float32) / 255.0

            # Build valid mask
            valid = np.ones_like(gray, dtype=bool)
            if ignore_transparent and alpha is not None:
                valid &= (alpha > 0)
            if ignore_white:
                # Re-evaluate whiteness on resized gray if desired
                valid &= (gray * 255.0 < float(white_thresh))
            imgs.append(gray)
            masks.append(valid)
        return np.stack(imgs, axis=0), np.stack(masks, axis=0)

    colmap_imgs, colmap_valid = load_and_preprocess(colmap_paths)
    orig_imgs,   orig_valid   = load_and_preprocess(orig_paths)

    N, H, W = colmap_imgs.shape
    M = orig_imgs.shape[0]

    indices = np.full((N,), -1, dtype=np.int64)
    best = np.full((N,), np.inf, dtype=np.float32)
    scores = np.full((N, M), np.nan, dtype=np.float32) if return_scores else None

    # Chunk over COLMAP images to keep memory bounded
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        A = colmap_imgs[s:e]              # (c,H,W)
        Am = colmap_valid[s:e]            # (c,H,W)
        # Expand orig one-by-one to keep peak memory low
        # Vectorized across the chunk for each orig image
        # MSE = sum(valid*(A-B)^2)/sum(valid)
        # where valid = Am & Bm
        for j in range(M):
            B  = orig_imgs[j]             # (H,W)
            Bm = orig_valid[j]            # (H,W)
            valid = Am & Bm[None, ...]    # (c,H,W)
            denom = valid.sum(axis=(1, 2)).astype(np.float32)  # (c,)
            if np.any(denom == 0):
                # Avoid divide-by-zero: set those pairs to inf
                mask_nonzero = denom > 0
            diff = A - B[None, ...]
            se = diff * diff
            se[~valid] = 0.0
            num = se.sum(axis=(1, 2))  # (c,)
            mse = np.full_like(num, np.inf)
            nz = denom > 0
            mse[nz] = num[nz] / denom[nz]
            if return_scores:
                scores[s:e, j] = mse
            # Update best for this chunk
            better = mse < best[s:e]
            indices[s:e][better] = j
            best[s:e][better] = mse[better]

    return indices, scores

# def _natural_sorted_paths(dirpath: str | Path, pattern: str = "*.png") -> List[Path]:
#     paths = list(Path(dirpath).glob(pattern))
#     def key(p: Path):
#         # sort by all integers in the stem, e.g. "r_100" -> (100,), "frame_00100" -> (100,)
#         nums = re.findall(r"\d+", p.stem)
#         return tuple(int(n) for n in nums) if nums else (p.stem,)
#     return sorted(paths, key=key)

# def match_by_mse_from_dirs(
#     colmap_dir: str | Path,
#     orig_dir: str | Path,
#     pattern: str = "*.png",
#     return_numeric_idx: bool = False,
#     **kwargs,
# ) -> np.ndarray:
#     """
#     Returns idx such that idx[i] is the best original image index for COLMAP image i.
#     If return_numeric_idx=True, idx values are the numeric suffix from filenames
#     (e.g., 'r_100' -> 100). Otherwise, idx are positions in the natural-sorted list.
#     """
#     colmap_paths = _natural_sorted_paths(colmap_dir, pattern)
#     orig_paths   = _natural_sorted_paths(orig_dir, pattern)

#     pos_idx, _ = brute_force_match_indices_by_mse(colmap_paths, orig_paths, **kwargs)

#     if return_numeric_idx:
#         # Map positions -> numeric ids extracted from the original filenames
#         orig_ids = np.array([int(re.findall(r"\d+", p.stem)[-1]) for p in orig_paths], dtype=np.int64)
#         return orig_ids[pos_idx]

#     return pos_idx

def SE3error(T, That):
    Terr = np.linalg.inv(T) @ That

    r = rot.from_matrix(Terr[:3, :3])
    axis_angle = r.as_rotvec()
    axis_angle = axis_angle / np.linalg.norm(axis_angle)

    rerr = abs(np.arccos(min(max(((Terr[0:3,0:3]).trace() - 1) / 2, -1.0), 1.0)))

    terr = np.linalg.norm(Terr[0:3,3])
    return (rerr*180/np.pi, terr, axis_angle[0], axis_angle[1], axis_angle[2])

def abs_orientation(X, Y):
    """
    Determine the optimal transformation that brings points from
    X's reference frame to points in Y's.
    T(x) = c * Rx + t where x is a point 3x1, c is the scaling, 
    R is a 3x3 rotation matrix, and t is a 3x1 translation.

    This is based off of "Least-Squares Estimation of Transformation
    Parameters Between Two Point Patterns" Umeyama 1991.

    Inputs:
        X - Tensor with dimension N x m
        Y - Tensor with dimension N x m
    Outputs:
        c - Scalar scaling constant
        R - Tensor 3x3 rotation matrix
        t - Tensor 3
    """

    # def umeyama_sim3(X, Y):
    # # X, Y: (N, 3)
    # N = X.shape[0]
    # mux = X.mean(0, keepdim=True)
    # muy = Y.mean(0, keepdim=True)
    # Xc = X - mux
    # Yc = Y - muy

    # sx2 = (Xc.pow(2).sum(dim=1).mean())  # variance of X
    # Sxy = (Yc.unsqueeze(2) @ Xc.unsqueeze(1)).mean(dim=0)  # 3x3

    # U, D, Vt = torch.linalg.svd(Sxy)
    # # Proper sign correction per Umeyama:
    # S = torch.eye(3, dtype=X.dtype, device=X.device)
    # if torch.det(U @ Vt) < 0:
    #     S[-1, -1] = -1.0

    # R = U @ S @ Vt
    # c = (torch.trace(torch.diag(D) @ S) / sx2).item()
    # t = (muy.T - c * (R @ mux.T)).squeeze()
    # return c, R, t

    # X, Y: (N, 3)
    # N = X.shape[0]
    # mux = X.mean(0, keepdim=True)
    # muy = Y.mean(0, keepdim=True)
    # Xc = X - mux
    # Yc = Y - muy

    # sx2 = (Xc.pow(2).sum(dim=1).mean())  # variance of X
    # Sxy = (Yc.unsqueeze(2) @ Xc.unsqueeze(1)).mean(dim=0)  # 3x3

    # U, D, Vt = torch.linalg.svd(Sxy)
    # # Proper sign correction per Umeyama:
    # S = torch.eye(3, dtype=X.dtype, device=X.device)
    # if torch.det(U @ Vt) < 0:
    #     S[-1, -1] = -1.0

    # R = U @ S @ Vt
    # c = (torch.trace(torch.diag(D) @ S) / sx2).item()
    # t = (muy.T - c * (R @ mux.T)).squeeze()
    # return c, R, t

    N, m = X.shape
    mux = torch.mean(X, 0, True)
    muy = torch.mean(Y, 0, True)
    
    Yd = (Y - muy).unsqueeze(-1)
    Xd = (X - mux).unsqueeze(1)
    sx = torch.sum(torch.norm(Xd.squeeze(), dim=1) ** 2) / N
    Sxy = (1 / N) * torch.sum(torch.matmul(Yd, Xd), dim=0)

    if torch.linalg.matrix_rank(Sxy) < m:
        raise NameError("Absolute orientation transformation does not exist!")

    U, D, Vt = torch.linalg.svd(Sxy, full_matrices=True)
    S = torch.eye(m).to(dtype=Vt.dtype)
    if torch.linalg.det(Sxy) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    c = torch.trace(torch.diag(D) @ S) / sx
    t = muy.T - c * (R @ mux.T)

    return c, R, t.squeeze()

data_dir = "traj9-13_2"
colmap_fp = f"{data_dir}/colmap/transforms.json"
mocap_fp = f"{data_dir}/transforms_ros.json"

colmap_img_dir = f"{data_dir}/colmap/images"
orig_img_dir = f"{data_dir}/images"

colmap_pcd_fp = f"{data_dir}/colmap/sparse_pc.ply"

save_fp = f"{data_dir}/transforms.json"
pcd_save_fp = f"{data_dir}/sparse_pc.ply"

with open(colmap_fp, 'r') as f:
    colmap_data = json.load(f)

with open(mocap_fp, 'r') as f:
    mocap_data = json.load(f)

colmap_sparse_pcd = o3d.io.read_point_cloud(colmap_pcd_fp)

colmap_frames = colmap_data["frames"]
mocap_frames = mocap_data["frames"]

colmap_transforms = torch.tensor([frame["transform_matrix"] for frame in colmap_frames])
mocap_transforms = torch.tensor([frame["transform_matrix"] for frame in mocap_frames])

colmap_images_paths = [f"{data_dir}/colmap/{frame['file_path']}" for frame in colmap_frames]
mocap_images_paths = [f"{data_dir}/{frame['file_path']}" for frame in mocap_frames]

assert len(colmap_frames) == len(mocap_frames), "Number of colmap and mocap frames must be the same"

idx, score = brute_force_match_indices_by_mse(colmap_images_paths, mocap_images_paths)

colmap_transforms_ = torch.eye(4)[None].repeat(len(colmap_transforms), 1, 1)

colmap_transforms_[idx] = colmap_transforms

colmap_transforms = colmap_transforms_

#%% 

colmap_pcd = torch.tensor(colmap_transforms)[:, :3, -1]
mocap_pcd = torch.tensor(mocap_transforms)[:, :3, -1]

c, R, t = abs_orientation(colmap_pcd, mocap_pcd)

colmap_in_mocap_pcd = c * torch.matmul(R, colmap_pcd.unsqueeze(-1)).squeeze() + t # (c*R @ colmap_pcd.T + t.unsqueeze(-1)).T
print(torch.linalg.norm(colmap_in_mocap_pcd - mocap_pcd, dim=-1))

c_pcd = o3d.geometry.PointCloud()
c_pcd.points = o3d.utility.Vector3dVector(colmap_in_mocap_pcd)
c_pcd.paint_uniform_color(np.array([1., 0., 0.]))

m_pcd = o3d.geometry.PointCloud()
m_pcd.points = o3d.utility.Vector3dVector(mocap_pcd)
m_pcd.paint_uniform_color(np.array([0., 1., 0.]))

o3d.visualization.draw_geometries([c_pcd, m_pcd])
#%%

colmap_in_mocap_transforms = torch.stack([torch.eye(4)]*len(colmap_transforms), dim=0)
colmap_in_mocap_transforms[:, :3, :3] = torch.matmul(R, colmap_transforms[:, :3, :3])
colmap_in_mocap_transforms[:, :3, 3] = c*torch.matmul(R, colmap_transforms[:, :3, 3].unsqueeze(-1)).squeeze() + t

errors = []
for c_t, m_t in zip(mocap_transforms.numpy(), colmap_in_mocap_transforms.numpy()):
    # m_t = np.concatenate([m_t, np.array([0, 0, 0, 1])[None,:]], axis=0)
    se3error = SE3error(c_t, m_t)
    print(se3error)
    errors.append(np.array(se3error))

errors = np.stack(errors)

print('Mean statistics: ', np.mean(errors, axis=0))
print('Std statistics: ', np.std(errors, axis=0))
print('Max statistics: ', np.max(errors, axis=0))
print('Min statistics: ', np.min(errors, axis=0))

#%% Returns new transforms file

colmap_in_mocap_data = colmap_data

old_frames = colmap_in_mocap_data['frames']

new_frames = []

og_mocap_paths = [f"{frame['file_path']}" for frame in mocap_frames]
# for i, frame_id in enumerate(idx):
for i in range(len(og_mocap_paths)):
    frame = {
        "file_path": og_mocap_paths[i],
        "transform_matrix": colmap_in_mocap_transforms[i].tolist(),
        # "colmap_im_id": frame_id+1
    }

    new_frames.append(frame)

colmap_in_mocap_data["frames"] = new_frames
colmap_in_mocap_data["applied_transform"] = np.eye(4)[:3].tolist()

with open(save_fp, 'w') as f:
    json.dump(colmap_in_mocap_data, f, indent=4)

#%%
colmap_pts = torch.tensor(colmap_sparse_pcd.points, dtype=torch.float32)

colmap_in_mocap_pcd = c * torch.matmul(R, colmap_pts.unsqueeze(-1)).squeeze() + t # (c*R @ colmap_pcd.T + t.unsqueeze(-1)).T

colmap_in_mocap_point_cloud = o3d.geometry.PointCloud()
colmap_in_mocap_point_cloud.points = o3d.utility.Vector3dVector(colmap_in_mocap_pcd)
colmap_in_mocap_point_cloud.colors = o3d.utility.Vector3dVector(colmap_sparse_pcd.colors)

o3d.io.write_point_cloud(pcd_save_fp, colmap_in_mocap_point_cloud)
