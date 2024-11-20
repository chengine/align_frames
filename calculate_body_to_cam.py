#%%
from scipy.spatial.transform import Rotation as R
import numpy as np

# parent": "body",
# 			"child":  "imu_apps",
# 			"T_child_wrt_parent": [0.0295, -0.0065, -0.016],
# 			"RPY_parent_to_child":    [0, 0, 0]

# "parent": "imu_apps",
# 			"child":  "tracking_front",
# 			"T_child_wrt_parent": [0.037, 0.000, 0.0006],
# 			"RPY_parent_to_child":    [0, 90, 90]

# "parent": "body",
# 			"child":  "tof",
# 			"T_child_wrt_parent": [0.066, 0.009, -0.012],
# 			"RPY_parent_to_child":    [0, 90, 180]

# "parent": "tracking_front",
# 			"child": "hires_front",
# 			"T_child_wrt_parent": [-0.00, 0.017, -.014],
# 			"RPY_parent_to_child":   [0, 0, 0]
# 		},

body_to_imu_apps = R.from_euler('XYZ', [0, 0, 0], degrees=True)
body_to_imu_apps = body_to_imu_apps.as_matrix()
body_to_imu_apps_trans = np.array([0.0295, -0.0065, -0.016])

body_to_imu_apps_transform = np.eye(4)
body_to_imu_apps_transform[:3, :3] = body_to_imu_apps
body_to_imu_apps_transform[:3, 3] = body_to_imu_apps_trans

imu_apps_to_tracking_front = R.from_euler('XYZ', [0, 90, 90], degrees=True)
imu_apps_to_tracking_front = imu_apps_to_tracking_front.as_matrix()
imu_apps_to_tracking_front_trans = np.array([0.037, 0.000, 0.0006])

imu_apps_to_tracking_front_transform = np.eye(4)
imu_apps_to_tracking_front_transform[:3, :3] = imu_apps_to_tracking_front
imu_apps_to_tracking_front_transform[:3, 3] = imu_apps_to_tracking_front_trans

tracking_front_to_hires_front = R.from_euler('XYZ', [0, 0, 0], degrees=True)
tracking_front_to_hires_front = tracking_front_to_hires_front.as_matrix()
tracking_front_to_hires_front_trans = np.array([-0.00, 0.017, -.014])

tracking_front_to_hires_front_transform = np.eye(4)
tracking_front_to_hires_front_transform[:3, :3] = tracking_front_to_hires_front
tracking_front_to_hires_front_transform[:3, 3] = tracking_front_to_hires_front_trans

body_to_tof = R.from_euler('XYZ', [0, 90, 180], degrees=True)
body_to_tof = body_to_tof.as_matrix()
body_to_tof_trans = np.array([0.066, 0.009, -0.012])

body_to_tof_transform = np.eye(4)
body_to_tof_transform[:3, :3] = body_to_tof
body_to_tof_transform[:3, 3] = body_to_tof_trans

body_to_hires_front_transform = tracking_front_to_hires_front_transform @ imu_apps_to_tracking_front_transform @ body_to_imu_apps_transform

body_to_hires_front_transform[:3, 3] = tracking_front_to_hires_front@tracking_front_to_hires_front_trans + imu_apps_to_tracking_front_trans + body_to_imu_apps_trans

print(body_to_hires_front_transform)
print(body_to_tof_transform)
#%%