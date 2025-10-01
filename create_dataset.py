#%%
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import cv2
from rosbags.image import message_to_cvimage
from scipy.spatial.transform import Rotation as R
import numpy as np
from sensor_msgs.msg import PointCloud2

import json
import open3d as o3d
from pathlib import Path

camera_to_body = np.array([[ 0.00000000e+00, -2.22044605e-16,  1.00000000e+00,  6.65000000e-02],
 [ 1.00000000e+00,  2.22044605e-16,  0.00000000e+00,  1.05000000e-02],
 [-2.22044605e-16,  1.00000000e+00,  2.22044605e-16, -2.94000000e-02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

opengl_to_opencv = np.array([ 
    [1., 0., 0., 0.],
    [0., -1., 0., 0.],
    [0., 0., -1., 0.],
    [0., 0., 0., 1.]
])

tof_to_body = np.array([[0.00000000e+00, -2.22044605e-16,  1.00000000e+00,  6.60000000e-02],
 [ 0,  -1.,  0.00000000e+00,  9.00000000e-03],
 [ 1,  0,  2.22044605e-16, -1.20000000e-02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

import struct
import math
from sensor_msgs.msg import PointCloud2, PointField
import sys

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ("b", 1)
_DATATYPES[PointField.UINT8] = ("B", 1)
_DATATYPES[PointField.INT16] = ("h", 2)
_DATATYPES[PointField.UINT16] = ("H", 2)
_DATATYPES[PointField.INT32] = ("i", 4)
_DATATYPES[PointField.UINT32] = ("I", 4)
_DATATYPES[PointField.FLOAT32] = ("f", 4)
_DATATYPES[PointField.FLOAT64] = ("d", 8)

def get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = ">" if is_bigendian else "<"
    offset = 0
    for field in (
        f
        for f in sorted(fields, key=lambda f: f.offset)
        if field_names is None or f.name in field_names
    ):
        if offset < field.offset:
            fmt += "x" * (field.offset - offset)
            offset = field.offset
        if field.datatype in _DATATYPES:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length
    return fmt

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    #assert isinstance(cloud, PointCloud2), "cloud is not a sensor_msgs.msg.PointCloud2"

    fmt = get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = (
        cloud.width,
        cloud.height,
        cloud.point_step,
        cloud.row_step,
        cloud.data,
        math.isnan,
    )
    
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step

save_folder = "traj9-13_2"
path = Path(save_folder).mkdir(parents=True, exist_ok=True)

# Create a typestore and get the string class.
typestore = get_typestore(Stores.LATEST)

bag_name = save_folder
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

        if connection.topic == '/republished_pose':
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            positions = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            quaternion = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

            rot_object = R.from_quat(quaternion)
            rot_mat = rot_object.as_matrix()

            transform = np.eye(4)
            transform[:3, :3] = rot_mat
            transform[:3, 3] = positions

            transform_c2w = transform @ camera_to_body @ opengl_to_opencv

            qvio_poses.append(transform_c2w)
            qvio_timestamps.append(timestamp)

        # if connection.topic == '/republished_pointcloud':
        #     msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
  
        #     point_cloud = np.array(list(read_points(msg)))
        #     #gen = pc2.read_points(msg, skip_nans=True)
        #     #int_data = list(gen)

        #     #point_cloud = np.array(int_data)

        #     point_cloud = point_cloud[~np.all(point_cloud == 0, axis=1)]

        #     point_clouds.append(point_cloud.T)
        #     point_cloud_timestamps.append(timestamp)

#%%

# Finds the temporally closest qvio pose and point cloud to each image timestamp

qvios = []
pcds = []

for timestamp in img_timestamps:
    idx = np.argmin(np.abs(np.array(qvio_timestamps) - timestamp))
    qvio_pose = qvio_poses[idx]
    qvios.append(qvio_pose)

    # idx = np.argmin(np.abs(np.array(point_cloud_timestamps) - timestamp))
    # pcd = point_clouds[idx]

    # transform = qvio_pose @ np.linalg.inv(opengl_to_opencv) @ np.linalg.inv(camera_to_body)
    # transform = qvio_pose @ np.linalg.inv(camera_to_body)

    # tof_transform = transform @ tof_to_body
    # point_cloud = tof_transform[:3, :3] @ pcd + tof_transform[:3, 3][:, None]

    # pcds.append(point_cloud.T)

qvios = np.array(qvios)

# for timestamp in point_cloud_timestamps:
#     idx = np.argmin(np.abs(np.array(qvio_timestamps) - timestamp))
#     qvio_pose = qvio_poses[idx]
#     qvios.append(qvio_pose)

#     idx = np.argmin(np.abs(np.array(point_cloud_timestamps) - timestamp))
#     pcd = point_clouds[idx]

#     transform = qvio_pose @ np.linalg.inv(opengl_to_opencv) @ np.linalg.inv(camera_to_body)

#     tof_transform = transform @ tof_to_body
#     point_cloud = tof_transform[:3, :3] @ pcd + tof_transform[:3, 3][:, None]

#     pcds.append(point_cloud.T)

# qvios = np.array(qvios)
#%%

# Saves the point cloud for Nerfstudio initialization
# pcds_all = np.concatenate(pcds, axis=0)

# pcds_viz = o3d.geometry.PointCloud()
# pcds_viz.points = o3d.utility.Vector3dVector(pcds_all)

# pcd_traj = o3d.geometry.PointCloud()
# pcd_traj.points = o3d.utility.Vector3dVector(qvios[:, :3, -1])
# pcd_traj.paint_uniform_color([1, 0, 1])

# o3d.visualization.draw_geometries([pcds_viz, pcd_traj])

# # Downsample pcd if necessary
# pcds_viz = pcds_viz.uniform_down_sample(100)
# print('Number of points:', len(pcds_viz.points))
# o3d.io.write_point_cloud(f"{save_folder}/sparse.ply", pcds_viz)
#%%

#NOTE: You might have to change these camera intrinsics

height, width = imgs[0].shape[:2]

# # Camera intrinsics
K = np.array([
    [5.0144869846790692e+02, 0., 5.0242559803716131e+02],
    [0., 5.0250709829431065e+02, 3.9420622264316472e+02],
    [0., 0., 1.]
])

d = np.array([-2.5632915666509198e-02, 1.7269314171904152e-02,
       -2.8638048363978191e-02, 1.3676976907675557e-02 ])

# K = np.array([[ 6.9821625652365788e+02, 0., 5.0056793248447093e+02],
#         [0., 6.9858447840373026e+02, 3.9642189572602103e+02],
#         [0., 0., 1. ]])

# d = np.array([ -3.4512907517987079e-01, 1.7894920785023438e-01,
#        -4.5397747876679676e-04, 1.2821427750666064e-05,
#        -5.9015795549746633e-02 ])

k1 = d[0]
k2 = d[1]
k3 = d[2]
k4 = 0.0
p1 = d[3]
p2 = 0.0#d[4]

# Which image index to start from. This is useful for when the lens first opens and the first few frames are dark.
start_idx = 0

path = Path(save_folder + '/images').mkdir(parents=True, exist_ok=True)

frames = []
for i, (img, pose) in enumerate(zip(imgs[start_idx:], qvios[start_idx:])):

    #NOTE: Uncomment this if you want to save the images
    cv2.imwrite(f'{save_folder}/images/r_{i}.png', img)

    frame = {
        "file_path": f'images/r_{i}.png',
        "transform_matrix": pose.tolist()
    }
    frames.append(frame)

qvio_data = {
    "w": width,
    "h": height,
    "fl_x": K[0, 0],
    "fl_y": K[1, 1],
    "cx": K[0, 2],
    "cy": K[1, 2],
    "k1": k1,
    "k2": k2,
    "k3": k3,
    "k4": k4,
    "p1": p1,
    "p2": p2,
    "camera_model": "OPENCV_FISHEYE",
    "frames": frames,
    "applied_transform": np.eye(4)[:3].tolist(),
    "ply_file_path": "sparse.ply"
}

transforms_path = f'{save_folder}/transforms.json'
with open(transforms_path, 'w') as f:
    json.dump(qvio_data, f, indent=4)

# %%
