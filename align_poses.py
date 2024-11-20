#%%
import json
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as rot

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

colmap_fp = 'data/splat/transforms.json'
mocap_fp = 'data/mocap/transforms.json'

with open(colmap_fp, 'r') as f:
    colmap_data = json.load(f)

with open(mocap_fp, 'r') as f:
    mocap_data = json.load(f)

colmap_sparse_pcd = o3d.io.read_point_cloud('data/splat/sparse_pc.ply')
mocap_sparse_pcd = o3d.io.read_point_cloud('data/mocap/points_color.ply')

colmap_frames = colmap_data["frames"]
mocap_frames = mocap_data["frames"]

colmap_transforms = torch.tensor([frame["transform_matrix"] for frame in colmap_frames])
mocap_transforms = torch.tensor([frame["transform_matrix"] for frame in mocap_frames])

colmap_frames_id = [frame["colmap_im_id"]-1 for frame in colmap_frames]

colmap_transforms_ = torch.ones_like(colmap_transforms)

colmap_transforms_[colmap_frames_id] = colmap_transforms

colmap_transforms = colmap_transforms_

# colmap_colmap_sparse_pcd = o3d.io.read_point_cloud('data/splat/sparse_pc_colmap.ply')
# o3d.visualization.draw_geometries([mocap_sparse_pcd])
# o3d.visualization.draw_geometries([colmap_sparse_pcd, mocap_sparse_pcd])

# mocap_sparse_pcd.colors = o3d.utility.Vector3dVector(np.flip(np.asarray(mocap_sparse_pcd.colors), axis=-1))
# o3d.io.write_point_cloud('data/splat_nav_new_nerfstudio/points_color_rectified.ply', mocap_sparse_pcd)
#%% 

colmap_pcd = torch.tensor(colmap_transforms)[:, :3, -1]
mocap_pcd = torch.tensor(mocap_transforms)[:, :3, -1]

c, R, t = abs_orientation(colmap_pcd, mocap_pcd)
# c, R, t = abs_orientation(colmap_pcd, mocap_pcd)

colmap_in_mocap_pcd = c * torch.matmul(R, colmap_pcd.unsqueeze(-1)).squeeze() + t # (c*R @ colmap_pcd.T + t.unsqueeze(-1)).T
print(torch.linalg.norm(colmap_in_mocap_pcd - mocap_pcd, dim=-1))

# mocap_in_colmap_pcd = c * torch.matmul(R, mocap_pcd.unsqueeze(-1)).squeeze() + t # (c*R @ mocap_pcd.T + t.unsqueeze(-1)).T
# print(torch.linalg.norm(mocap_in_colmap_pcd - colmap_pcd, dim=-1))

# c_pcd = o3d.geometry.PointCloud()
# c_pcd.points = o3d.utility.Vector3dVector(colmap_in_mocap_pcd)
# c_pcd.paint_uniform_color(np.array([1., 0., 0.]))

# m_pcd = o3d.geometry.PointCloud()
# m_pcd.points = o3d.utility.Vector3dVector(mocap_pcd)
# m_pcd.paint_uniform_color(np.array([0., 1., 0.]))

# o3d.visualization.draw_geometries([c_pcd, m_pcd])
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

for i, frame_id in enumerate(colmap_frames_id):

    frame = {
        "file_path": old_frames[i]["file_path"],
        "transform_matrix": colmap_in_mocap_transforms[frame_id].tolist(),
        "colmap_im_id": frame_id+1
    }

    new_frames.append(frame)

colmap_in_mocap_data["frames"] = new_frames
colmap_in_mocap_data["applied_transform"] = np.eye(4)[:3].tolist()

with open('data/colmap_in_mocap/transforms.json', 'w') as f:
    json.dump(colmap_in_mocap_data, f, indent=4)

#%%
colmap_pts = torch.tensor(colmap_sparse_pcd.points, dtype=torch.float32)

colmap_in_mocap_pcd = c * torch.matmul(R, colmap_pts.unsqueeze(-1)).squeeze() + t # (c*R @ colmap_pcd.T + t.unsqueeze(-1)).T

colmap_in_mocap_point_cloud = o3d.geometry.PointCloud()
colmap_in_mocap_point_cloud.points = o3d.utility.Vector3dVector(colmap_in_mocap_pcd)
colmap_in_mocap_point_cloud.colors = o3d.utility.Vector3dVector(colmap_sparse_pcd.colors)

with open('gaussians.json', 'r') as fp:
    meta = json.load(fp)

means = np.array(meta['means'])
colors = np.array(meta['colors'])

trained_colmap_in_mocap = o3d.geometry.PointCloud()
trained_colmap_in_mocap.points = o3d.utility.Vector3dVector(means)
trained_colmap_in_mocap.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([colmap_in_mocap_point_cloud, trained_colmap_in_mocap])
# o3d.io.write_point_cloud('data/splat_nav_new_nerfstudio/points_colmap.ply', colmap_in_mocap_point_cloud)
#%% 

# colmap_pts = torch.tensor(colmap_sparse_pcd.points, dtype=torch.float32)

# colmap_in_mocap_pcd = c * torch.matmul(R, colmap_pts.unsqueeze(-1)).squeeze() + t # (c*R @ colmap_pcd.T + t.unsqueeze(-1)).T

# colmap_in_mocap_point_cloud = o3d.geometry.PointCloud()
# colmap_in_mocap_point_cloud.points = o3d.utility.Vector3dVector(colmap_in_mocap_pcd)
# colmap_in_mocap_point_cloud.colors = o3d.utility.Vector3dVector(colmap_sparse_pcd.colors)
# o3d.io.write_point_cloud('data/splat_nav_new_nerfstudio/points_colmap.ply', colmap_in_mocap_point_cloud)
# %%

# mocap_pts = torch.tensor(mocap_sparse_pcd.points, dtype=torch.float32)
# mocap_in_colmap_pcd = c * torch.matmul(R, mocap_pts.unsqueeze(-1)).squeeze() + t # (c*R @ colmap_pcd.T + t.unsqueeze(-1)).T

# mocap_in_colmap_point_cloud = o3d.geometry.PointCloud()
# mocap_in_colmap_point_cloud.points = o3d.utility.Vector3dVector(mocap_in_colmap_pcd)
# mocap_in_colmap_point_cloud.colors = o3d.utility.Vector3dVector(mocap_sparse_pcd.colors)
# o3d.io.write_point_cloud('data/splat/sparse_pc_mocap.ply', mocap_in_colmap_point_cloud)