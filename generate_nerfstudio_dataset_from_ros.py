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
        if connection.topic == '/republished_image':
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            img = message_to_cvimage(msg, 'bgr8')

            imgs.append(img)
            img_timestamps.append(timestamp)

        if connection.topic == '/vrpn_mocap/modal_ai/pose':
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
 
            positions = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            quaternion = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

            rot_object = R.from_quat(quaternion)
            rot_mat = rot_object.as_matrix()

            transform = np.eye(4)
            transform[:3, :3] = rot_mat
            transform[:3, 3] = positions

            mocap_poses.append(transform)
            mocap_timestamps.append(timestamp)

        if connection.topic == '/republished_pose':
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            positions = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            quaternion = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

            rot_object = R.from_quat(quaternion)
            rot_mat = rot_object.as_matrix()

            transform = np.eye(4)
            transform[:3, :3] = rot_mat
            transform[:3, 3] = positions

            qvio_poses.append(transform)
            qvio_timestamps.append(timestamp)

        if connection.topic == '/republished_pointcloud':
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
  
            gen = pc2.read_points(msg, skip_nans=True)
            int_data = list(gen)

            point_cloud = np.array(int_data)

            point_cloud = point_cloud[~np.all(point_cloud == 0, axis=1)]

            point_clouds.append(point_cloud)
            point_cloud_timestamps.append(timestamp)

# %%
# fig, ax = plt.subplots(1, figsize=(10, 10))

path = Path(f"images_{bag_name}").mkdir(parents=True, exist_ok=True)

for i, (img, timestamp) in enumerate(zip(imgs, img_timestamps)):
    # ax.imshow(img)
    # plt.show()
    # ax.set_title(f"Timestamp: {timestamp}")

    cv2.imwrite(f'images_{bag_name}/r_{i}.png', img)

# %%

qvio_poses_save = []
mocap_poses_save = []
point_clouds_save = []

for timestamp in img_timestamps:

    if len(mocap_timestamps) > 0:
        idx = np.argmin(np.abs(np.array(mocap_timestamps) - timestamp))

        mocap_pose = mocap_poses[idx]
        mocap_poses_save.append(mocap_pose.tolist())

    idx = np.argmin(np.abs(np.array(qvio_timestamps) - timestamp))

    qvio_pose = qvio_poses[idx]

    idx = np.argmin(np.abs(np.array(point_cloud_timestamps) - timestamp))

    point_cloud = point_clouds[idx]
    transformed_point_cloud = point_cloud.T #qvio_pose[:3, :3] @ point_cloud.T + qvio_pose[:3, :3].T @qvio_pose[:3, 3][:, None]

    qvio_poses_save.append(qvio_pose.tolist())
    point_clouds_save.append(transformed_point_cloud.T)

pcds_np = np.concatenate(point_clouds_save, axis=0)

#%%
pcds = o3d.geometry.PointCloud()
pcds.points = o3d.utility.Vector3dVector(pcds_np)
o3d.visualization.draw_geometries([pcds])

#%%
# # Find relative transform between two point clouds
# pcd1 = o3d.geometry.PointCloud()
# pcd1.points = o3d.utility.Vector3dVector(point_clouds_save[50])

# pcd2 = o3d.geometry.PointCloud()
# pcd2.points = o3d.utility.Vector3dVector(point_clouds_save[53])

# reg_p2p = o3d.pipelines.registration.registration_icp(
#     pcd1, pcd2, 0.001, np.eye(4),
#     o3d.pipelines.registration.TransformationEstimationPointToPoint())
# print(reg_p2p)
# print("Transformation is:")
# print(reg_p2p.transformation)
# pcd1_transformed = pcd1.transform(reg_p2p.transformation)

# o3d.visualization.draw_geometries([pcd2, pcd1_transformed])

#%%

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

qvio_pcd = torch.tensor(qvio_poses_save)[:, :3, -1]
mocap_pcd = torch.tensor(mocap_poses_save)[:, :3, -1]

c, R, t = abs_orientation(qvio_pcd, mocap_pcd)

qvio_in_mocap_pcd = c * torch.matmul(R, qvio_pcd.unsqueeze(-1)).squeeze() + t # (c*R @ colmap_pcd.T + t.unsqueeze(-1)).T
print(torch.linalg.norm(qvio_in_mocap_pcd - mocap_pcd, dim=-1))

qvio_pcd_full = c * torch.matmul(R, torch.tensor(qvio_poses, dtype=torch.float32)[:, :3, -1].unsqueeze(-1)).squeeze() + t 
mocap_pcd_full = torch.tensor(mocap_poses, dtype=torch.float32)[:, :3, -1]

c_pcd = o3d.geometry.PointCloud()
c_pcd.points = o3d.utility.Vector3dVector(qvio_pcd_full)
c_pcd.paint_uniform_color(np.array([1., 0., 0.]))

m_pcd = o3d.geometry.PointCloud()
m_pcd.points = o3d.utility.Vector3dVector(mocap_pcd_full)
m_pcd.paint_uniform_color(np.array([0., 1., 0.]))

meshes = [c_pcd, m_pcd]

for i in range(0, len(qvio_pcd_full)):
    if i % 10 == 0:
        origin = qvio_pcd_full[i]
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin = origin)
        axes.rotate(np.array(qvio_poses[i])[:3, :3], center=origin)
        meshes.append(axes)
        break

for i in range(0, len(mocap_pcd_full)):
    if i % 100 == 0:
        origin = mocap_pcd_full[i]
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin = origin)
        axes.rotate(np.array(mocap_poses[i])[:3, :3], center=origin)
        meshes.append(axes)
        break

o3d.visualization.draw_geometries(meshes)

#%%

qvio_transforms = torch.tensor(qvio_poses_save)
mocap_transforms = torch.tensor(mocap_poses_save)

rectified_rotation = torch.tensor(mocap_poses_save[0])[:3, :3] @ torch.linalg.inv(torch.tensor(qvio_poses_save[0])[:3, :3])

qvio_pcd_in_mocap_transforms = torch.stack([torch.eye(4)]*len(qvio_transforms), dim=0)
qvio_pcd_in_mocap_transforms[:, :3, :3] = torch.matmul(rectified_rotation, qvio_transforms[:, :3, :3])
qvio_pcd_in_mocap_transforms[:, :3, 3] = c*torch.matmul(R, qvio_transforms[:, :3, 3].unsqueeze(-1)).squeeze() + t

errors = []
for c_t, m_t in zip(mocap_transforms.numpy(), qvio_pcd_in_mocap_transforms.numpy()):
    # m_t = np.concatenate([m_t, np.array([0, 0, 0, 1])[None,:]], axis=0)
    se3error = SE3error(c_t, m_t)
    print(se3error)
    errors.append(np.array(se3error))

errors = np.stack(errors)

print('Mean statistics: ', np.mean(errors, axis=0))
print('Std statistics: ', np.std(errors, axis=0))
print('Max statistics: ', np.max(errors, axis=0))
print('Min statistics: ', np.min(errors, axis=0))
# %%
