import numpy as np
import torch
import cv2
import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
import os
from model import TCP
from data import CARLA_Data
from config import GlobalConfig
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def draw_trajectory_on_ax(ax: Axes, trajectories, labels, colors, line_type='o-', transparent=True, xlim=(-70, 70), ylim=(0, 120)):
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

def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
	updated_intrinsics = intrinsics.copy()
	# Adjust intrinsics scale due to resizing
	updated_intrinsics[0, 0] *= scale_width
	updated_intrinsics[0, 2] *= scale_width
	updated_intrinsics[1, 1] *= scale_height
	updated_intrinsics[1, 2] *= scale_height

	# Adjust principal point due to cropping
	updated_intrinsics[0, 2] -= left_crop
	updated_intrinsics[1, 2] -= top_crop

	return updated_intrinsics

def cal_camera_intrinsic(fov, update = False):
	f = 900 /(2 * np.tan(fov * np.pi/ 360))
	Cu = 900 / 2
	Cv = 256 / 2
	K = np.array([[f, 0, Cu],
	 [0, f, Cv],
	 [0, 0, 1 ]])
	if update:
		top_crop = ORIGIN_H//2 - 256//2
		left_crop = ORIGIN_W//2 - 256//2
		K = update_intrinsics(K, top_crop, left_crop)
	return K


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


if __name__ == "__main__":
	config = GlobalConfig()
	val_set = CARLA_Data(root=config.train_data)
	dataloader_val = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=8)

	net = TCP(config)
	ckpt = torch.load('/home/wupenghao/transfuser/TCP/log/40k_half_cnnatt/epoch=59-last.ckpt')
	ckpt = ckpt["state_dict"]
	new_state_dict = OrderedDict()
	for key, value in ckpt.items():
		new_key = key.replace("model.","")
		new_state_dict[new_key] = value
	net.load_state_dict(new_state_dict, strict = False)
	net.cuda()
	net.eval()

	camera_intrinsic = cal_camera_intrinsic(100)
	camera_extrinsic, camera_translation_inv, camera_rotation_matrix_inv = cal_camera_extrinsic(-1.5, 0, 2.0, -np.pi/2,0,-np.pi/2)
	total_number = 0
	for b_idx, batch in tqdm.tqdm(enumerate(dataloader_val)):
		if total_number > 300:
			break
		front_img = batch['front_img']
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']

		gt_waypoints = batch['waypoints']
		
		state = torch.cat([speed, target_point, command], 1)

		with torch.no_grad():
			pred, vis_list = net(front_img.cuda(), state.cuda(), target_point.cuda(), True)

			pred_wp = pred['pred_wp']
			gt_waypoints = gt_waypoints.numpy()


			gt_waypoints = -1 * gt_waypoints[:,:, ::-1]

			pred_wp = pred_wp.cpu().numpy()

			pred_wp = -1 * pred_wp[:,:, ::-1]

			mean = torch.zeros_like(front_img)
			mean[:,0,:,:] = .485
			mean[:,1,:,:] = .456
			mean[:,2,:,:] = .406
			std = torch.zeros_like(front_img)
			std[:,0,:,:] = 0.229
			std[:,1,:,:] = 0.224
			std[:,2,:,:] = 0.225
			front_img = (front_img*std + mean)*255
			front_img = front_img.numpy().transpose(0,2,3,1).astype(np.uint8)
			front_img = front_img[0]


			fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))

			ax1.imshow(front_img)
			ax1.set_title('image')

			trajectories = list(pred_wp) + list(gt_waypoints)
			labels = ["pred", "gt"]
			colors = ["r", "b"]


			ax2.imshow(front_img)
			trajectories = [np.concatenate([trajectory, np.zeros((4,1))], 1) for trajectory in trajectories]

			for trajectory_single, label, color in zip(trajectories, labels,colors):
				trajectory_single = trajectory_single[(trajectory_single[..., 0] > 0)]
				if len(trajectory_single) == 0:
					continue

				location = list((p + camera_translation_inv for p in trajectory_single))
				proj_trajectory = np.array(list((camera_intrinsic @ (camera_rotation_matrix_inv @ l) for l in location)))
				proj_trajectory /= proj_trajectory[..., 2:3].repeat(3, -1)

				proj_trajectory = proj_trajectory[(proj_trajectory[..., 0] > 0) & (proj_trajectory[..., 0] < 900)]
				proj_trajectory = proj_trajectory[(proj_trajectory[..., 1] > 0) & (proj_trajectory[..., 1] < 256)]
				ax2.plot(proj_trajectory[:, 0], proj_trajectory[:, 1], 'o-', label=label ,color = color)


			ax2.legend()
			plt.tight_layout()
			output_folder = "vis"
			if not os.path.exists(output_folder):
				os.mkdir(output_folder)
			plt.savefig(os.path.join(output_folder,"%04d.png" % total_number))
			total_number += 1
			plt.close(fig)

			att_vis_list = []
			for vis in vis_list:
				vis = vis[0].permute(1,2,0).repeat(1,1,3).cpu().numpy() * 255
				vis = vis.astype(np.uint8)
				vis = cv2.resize(vis, (900,256))
				vis = cv2.addWeighted(vis, 0.7, front_img, 1 - 0.7, 0)
				att_vis_list.append(vis)
			att_vis = np.vstack(att_vis_list)
			output_folder = "vis_att"
			if not os.path.exists(output_folder):
				os.mkdir(output_folder)
			cv2.imwrite(os.path.join(output_folder,"%04d.png" % total_number), att_vis)






# H = 8
# W = 29
# sigma = 0.08
# # mesh grid 
# xx = torch.arange(0, W).view(1,-1).repeat(H,1)
# yy = torch.arange(0, H).view(-1,1).repeat(1,W)
# xx = xx.view(1,H,W)
# yy = yy.view(1,H,W)
# grid = torch.cat((yy,xx),0).float()

# distance_map = torch.zeros([H,W,H,W])
# for i in range(H):
#     for j in range(W):
#         current_position = torch.tensor([i,j]).view(2,1,1)
#         square_distance = torch.pow((current_position - grid)[0]/H, 2) + torch.pow((current_position - grid)[1]/W, 2)
#         weight = torch.exp(-0.5*square_distance/(sigma**2))
#         distance_map[i][j] = weight
# distance_map /= torch.sum(distance_map, dim=(2,3))

# vis_map = 255*distance_map[7,15].view(H,W,1).repeat(1,1,3).numpy()
# vis_map = vis_map.astype(np.uint8)
# image = cv2.imread("0080.png")
# vis_map = cv2.resize(vis_map, (900,256))
# vis_map = cv2.addWeighted(vis_map, 0.8, image, 1 - 0.8, 0)



# camera_intrinsic = cal_camera_intrinsic(100)
# camera_extrinsic, camera_translation_inv, camera_rotation_matrix_inv = cal_camera_extrinsic(-8, 0, 2.0, -np.pi/2,0,-np.pi/2)

# wp = np.array([0, 0, 0])
# location = wp + camera_translation_inv
# location = camera_intrinsic@(camera_rotation_matrix_inv @ location)
# location = location/location[-1]
# print(location)

# vis_map[int(location[1]), int(location[0])] = 0
# cv2.imwrite("vis.png", vis_map)