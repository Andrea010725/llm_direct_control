from turtle import left
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torchvision.transforms as T

# length = 10

# img = cv2.imread('lane_change_data/Accident/Accident_10_10_15_52_31_RouteScenario_0_rep0/rgb_high/0040.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# with open('lane_change_data/Accident/Accident_10_10_15_52_31_RouteScenario_0_rep0/info/0040.json') as f:
#     info = json.load(f)
#     wp_list = info['planned_route'][:length]
#     ego_x = info['pos_x']
#     ego_y = info['pos_y']
#     theta = info['yaw']

# # convert to ego coordinate
# # x: right, y: downwards
# waypoints = []
# for i in range(length):
#     R = np.array([
#     [np.cos(np.pi/2+theta), -np.sin(np.pi/2+theta)],
#     [np.sin(np.pi/2+theta),  np.cos(np.pi/2+theta)]
#     ])
#     local_point = np.array([wp_list[i][1]-ego_y, wp_list[i][0]-ego_x] )
#     local_point = R.T.dot(local_point)
#     waypoints.append(local_point)

# waypoints = np.array(waypoints)


ORIGIN_W = 800
ORIGIN_H = 386

def euler_to_rotMat(roll, pitch, yaw):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])
    # R = RzRyRx
    rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    return rotMat

def cal_camera_extrinsic(x,y,z, roll, pitch, yaw):
    camera_rotation_matrix = euler_to_rotMat(roll, pitch, yaw)
    camera_translation = np.array([x, y, z])
    camera_extrinsic = np.vstack((np.hstack((camera_rotation_matrix, camera_translation.reshape((3, 1)))), np.array([0, 0, 0, 1])))
    camera_extrinsic = np.linalg.inv(camera_extrinsic)


    camera_translation_inv = -camera_translation
    camera_rotation_matrix_inv = np.linalg.inv(camera_rotation_matrix)

    return camera_extrinsic, camera_translation_inv, camera_rotation_matrix_inv


def cal_camera_intrinsic(fov):
    f = ORIGIN_W /(2 * np.tan(fov * np.pi/ 360))
    Cu = ORIGIN_W / 2
    Cv = ORIGIN_H / 2
    K = np.array([[f, 0, Cu],
     [0, f, Cv],
     [0, 0, 1 ]])
    return K

def draw_trajectory_on_ax(ax: Axes, trajectories, labels, colors, line_type='o-', transparent=True, xlim=(-20, 20), ylim=(0, 50)):
    '''
    ax: matplotlib.axes.Axes, the axis to draw trajectories on
    trajectories: List of numpy arrays of shape (num_points, 2 or 3)
    confs: List of numbers, 1 means gt
    '''
    for idx, (trajectory, label, color) in enumerate(zip(trajectories, labels, colors)):
        ax.plot(-1*trajectory[:, 1], trajectory[:, 0], line_type, label=label, color = color)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend()

    return ax

# extrinsics, intrincs
# camera_intrinsic = cal_camera_intrinsic(100)
# camera_extrinsic, camera_translation_inv, camera_rotation_matrix_inv = cal_camera_extrinsic(-0.5, 0, 2.4, -np.pi/2,0,-np.pi/2)


# waypoints = waypoints[:, ::-1]
# waypoints *= -1
# pred = waypoints



# fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))


# trajectories = [pred] + [waypoints]

# labels = ["pred", "gt"]
# colors = ["r", "b"]
# ax1 = draw_trajectory_on_ax(ax1, trajectories, labels, colors)

# ax2.imshow(img)
# trajectories = [np.concatenate([trajectory, np.zeros((length, 1))], 1) for trajectory in trajectories]


# for trajectory_single, label, color in zip(trajectories, labels, colors):
#     trajectory_single = trajectory_single[(trajectory_single[..., 0] > 0)]
#     if len(trajectory_single) == 0:
#         continue
#     location = list((p + camera_translation_inv for p in trajectory_single))
#     proj_trajectory = np.array(list((camera_intrinsic @ (camera_rotation_matrix_inv @ l) for l in location)))
#     proj_trajectory /= proj_trajectory[..., 2:3].repeat(3, -1)
#     proj_trajectory = proj_trajectory[(proj_trajectory[..., 0] > 0) & (proj_trajectory[..., 0] < 800)]
#     proj_trajectory = proj_trajectory[(proj_trajectory[..., 1] > 0) & (proj_trajectory[..., 1] < 386)]
#     ax2.plot(proj_trajectory[:, 0], proj_trajectory[:, 1], 'o-', label=label, color=color)

# ax2.legend()
# plt.tight_layout()
# plt.savefig('vis_waypoint1.png')
# plt.show()




def vis(class_id, wp_pre, wp_gt, img):
    camera_intrinsic = cal_camera_intrinsic(100)
    camera_extrinsic, camera_translation_inv, camera_rotation_matrix_inv = cal_camera_extrinsic(-0.5, 0, 2.4, -np.pi/2,0,-np.pi/2)
    
    waypoints = wp_gt.cpu().numpy()[:, ::-1]
    waypoints *= -1
    pred = -1*wp_pre.cpu().numpy()[:, ::-1]



    fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))


    trajectories = [pred] + [waypoints]
    print(trajectories)
    labels = ["pred", "gt"]
    colors = ["r", "b"]
    ax1 = draw_trajectory_on_ax(ax1, trajectories, labels, colors)
    img = img.cpu()
    transform = T.ToPILImage()
    img = transform(img)
    ax2.imshow(img)
    trajectories = [np.concatenate([trajectory, np.zeros((30, 1))], 1) for trajectory in trajectories]


    for trajectory_single, label, color in zip(trajectories, labels, colors):
        trajectory_single = trajectory_single[(trajectory_single[..., 0] > 0)]
        if len(trajectory_single) == 0:
            continue
        location = list((p + camera_translation_inv for p in trajectory_single))
        proj_trajectory = np.array(list((camera_intrinsic @ (camera_rotation_matrix_inv @ l) for l in location)))
        proj_trajectory /= proj_trajectory[..., 2:3].repeat(3, -1)
        proj_trajectory = proj_trajectory[(proj_trajectory[..., 0] > 0) & (proj_trajectory[..., 0] < 800)]
        proj_trajectory = proj_trajectory[(proj_trajectory[..., 1] > 0) & (proj_trajectory[..., 1] < 386)]
        ax2.plot(proj_trajectory[:, 0], proj_trajectory[:, 1], 'o-', label=label, color=color)
    # classes_dict = {'left_img_t': 0, 'neg_data': 1, 'right_img_t': 2}
    if class_id == 0:
        flag_position = "left"
    elif class_id == 2:
        flag_position = "right"
    else:
        flag_position = "no option"
    print(flag_position)
    ax2.legend()
    plt.title(flag_position)
    plt.tight_layout()
    plt.savefig(flag_position+'.png')
    plt.show()