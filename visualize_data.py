#%%
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import cv2
from rosbags.image import message_to_cvimage
from scipy.spatial.transform import Rotation as R
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import struct
import ctypes

import torch
from scipy.spatial.transform import Rotation as Rot
import json
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path


def SE3error(T, That):
    Terr = np.linalg.inv(T) @ That

    r = Rot.from_matrix(Terr[:3, :3])
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

camera_to_body = np.array([[ 0.00000000e+00, -2.22044605e-16,  1.00000000e+00,  6.65000000e-02],
 [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00,  1.05000000e-02],
 [-2.22044605e-16,  1.00000000e+00,  2.22044605e-16, -2.94000000e-02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# Create a typestore and get the string class.
typestore = get_typestore(Stores.LATEST)

bag_name = 'bag8'
# Create reader instance and open for reading.
with Reader(bag_name) as reader:
    # Topic and msgtype information is available on .connections list.
    for connection in reader.connections:
        print(connection.topic, connection.msgtype)

    # Iterate over messages.
    imgs = []
    img_timestamps = []

    mocap_poses = []
    mocap_timestamps = []

    qvio_poses = []
    qvio_timestamps = []

    point_clouds = []
    point_cloud_timestamps = []

    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == '/hires_small_color':
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            img = message_to_cvimage(msg, 'bgr8')

            imgs.append(img)
            img_timestamps.append(timestamp)

        if connection.topic == '/qvio_pose':
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            positions = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            quaternion = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

            rot_object = R.from_quat(quaternion)
            rot_mat = rot_object.as_matrix()

            transform = np.eye(4)
            transform[:3, :3] = rot_mat
            transform[:3, 3] = positions

            transform = transform @ camera_to_body

            qvio_poses.append(transform)
            qvio_timestamps.append(timestamp)

        if connection.topic == '/tof_pc':
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
  
            gen = pc2.read_points(msg, skip_nans=True)
            int_data = list(gen)

            point_cloud = np.array(int_data)

            point_cloud = point_cloud[~np.all(point_cloud == 0, axis=1)]

            point_clouds.append(point_cloud)
            point_cloud_timestamps.append(timestamp)

#%%


path = Path(f"images_{bag_name}").mkdir(parents=True, exist_ok=True)

for i, (img, timestamp) in enumerate(zip(imgs, img_timestamps)):

    cv2.imwrite(f'images_{bag_name}/r_{i}.png', img)

qvios = []
for timestamp in img_timestamps:
    idx = np.argmin(np.abs(np.array(qvio_timestamps) - timestamp))
    qvio_pose = qvio_poses[idx]
    qvios.append(qvio_pose)

qvios = np.array(qvios)[400:]

translations = qvios[:, :3, -1]

#%% Load colmap
# with open('transforms/transforms.json', 'r') as f:
#     colmap_data = json.load(f)

# colmap_frames = colmap_data["frames"]
# colmap_transforms = np.array([frame["transform_matrix"] for frame in colmap_frames])
# colmap_ids = np.array([frame["colmap_im_id"] for frame in colmap_frames])

# sorted_ids = np.argsort(colmap_ids)
# colmap_transforms = colmap_transforms[sorted_ids]
# colmap_translations = colmap_transforms[:, :3, -1]
#%%

jet = plt.get_cmap('jet') 

colors_qvio = jet(np.linspace(0, 1, len(translations)))[:, :3]
# colors_colmap = jet(np.linspace(0, 1, len(colmap_translations)))[:, :3]

q_pcd = o3d.geometry.PointCloud()
q_pcd.points = o3d.utility.Vector3dVector(translations)
q_pcd.colors = o3d.utility.Vector3dVector(colors_qvio)

# c_pcd = o3d.geometry.PointCloud()
# c_pcd.points = o3d.utility.Vector3dVector(colmap_translations)
# c_pcd.colors = o3d.utility.Vector3dVector(colors_colmap)

# line_set = o3d.geometry.LineSet()
# line_set = line_set.create_from_point_cloud_correspondences(q_pcd, c_pcd, [[i, i] for i in range(0, len(translations))])

meshes = [q_pcd]

for i in range(0, len(translations)):
    if i % 10 == 0:
        origin = translations[i]
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin = origin)
        axes.rotate(np.array(qvios[i])[:3, :3], center=origin)
        meshes.append(axes)

# for i in range(0, len(colmap_translations)):
#     if i % 10 == 0:
#         origin = colmap_translations[i]
#         axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin = origin)
#         axes.rotate(np.array(colmap_transforms[i])[:3, :3], center=origin)
#         meshes.append(axes)

# world_origin = np.zeros(3)
# world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin = world_origin)
# meshes.append(world_axes)

# colmap_start = colmap_translations[0]
# colmap_start = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin = colmap_start)
# meshes.append(colmap_start)

o3d.visualization.draw_geometries(meshes)

#%%

colmap_pcd = torch.tensor(colmap_translations)
qvio_pcd = torch.tensor(qvios[:, :3, -1])

c, R, t = abs_orientation(qvio_pcd, colmap_pcd)
qvio_in_colmap_pcd = c * torch.matmul(R, qvio_pcd.unsqueeze(-1)).squeeze() + t 
print(torch.linalg.norm(qvio_in_colmap_pcd - colmap_pcd, dim=-1))

q_pcd = o3d.geometry.PointCloud()
q_pcd.points = o3d.utility.Vector3dVector(qvio_in_colmap_pcd)
q_pcd.paint_uniform_color(np.array([1., 0., 0.]))

c_pcd = o3d.geometry.PointCloud()
c_pcd.points = o3d.utility.Vector3dVector(colmap_translations)
c_pcd.paint_uniform_color(np.array([0., 1., 0.]))

o3d.visualization.draw_geometries([q_pcd, c_pcd])

# qvio_in_colmap_transforms = torch.stack([torch.eye(4)]*len(qvio_pcd), dim=0)
# qvio_in_colmap_transforms[:, :3, :3] = torch.matmul(R, poses[:, :3, :3])
# qvio_in_colmap_transforms[:, :3, 3] = c*torch.matmul(R, poses[:, :3, 3].unsqueeze(-1)).squeeze() + t

# errors = []
# for c_t, m_t in zip(colmap_transforms.numpy(), qvio_in_colmap_transforms.numpy()):
#     se3error = SE3error(c_t, m_t)
#     print(se3error)
#     errors.append(np.array(se3error))

# errors = np.stack(errors)

# print('Mean statistics: ', np.mean(errors, axis=0))
# print('Std statistics: ', np.std(errors, axis=0))
# print('Max statistics: ', np.max(errors, axis=0))
# print('Min statistics: ', np.min(errors, axis=0))

#%%