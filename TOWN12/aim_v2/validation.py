import argparse
import os
import tqdm
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from train import AIM_planner
from data import CARLA_Data
from config import GlobalConfig

ORIGIN_W = 800
ORIGIN_H = 384

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
	f = ORIGIN_W /(2 * np.tan(fov * np.pi/ 360))
	Cu = ORIGIN_W / 2
	Cv = ORIGIN_H / 2
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


LEFT = cal_camera_extrinsic(0,0,0,-np.pi/2,0,-np.pi/2 + np.pi/3)
FRONT = cal_camera_extrinsic(0,0,0,-np.pi/2,0,-np.pi/2)
RIGHT = cal_camera_extrinsic(0,0,0,-np.pi/2,0,-np.pi/2 - np.pi/3)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--id', type=str, default='aim_recurrent', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=101, help='Number of train epochs.')
	parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
	parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)
	# Config
	config = GlobalConfig()
	# Data
	val_set = CARLA_Data(root = '/home/wupenghao/transfuser/data_roach_90/', data_folders =config.val_data)
	dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8)
	print(len(dataloader_val))
	AIM = AIM_planner(config = config, lr = args.lr, device ="cuda")
	AIM = AIM_planner.load_from_checkpoint("/home/wupenghao/transfuser/aim_v2/log/lb2_90routes_rgbhigh_half/epoch=59-last.ckpt", config = config, lr = args.lr, device ="cuda")

	AIM.eval()
	AIM.model.cuda()
	AIM.freeze()
	total_number = 0

	camera_intrinsic = cal_camera_intrinsic(100, False)
	camera_extrinsic, camera_translation_inv, camera_rotation_matrix_inv = cal_camera_extrinsic(-0.5, 0, 2.4, -np.pi/2,0,-np.pi/2)

	for b_idx, batch in tqdm.tqdm(enumerate(dataloader_val)):

		front_img = batch['front_img'].cuda()
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command'].to(dtype=torch.float32)
		state = torch.cat([speed, target_point, command], 1)


		gt_waypoints = batch['waypoints']
		value = batch['value'].view(-1,1)
		feature = batch['feature']
		# inference
		pred_wp = AIM.model(front_img, state.cuda(), target_point.cuda())['pred_wp']
		

		gt_waypoints = gt_waypoints.cpu().numpy()

		gt_waypoints = gt_waypoints[:,:, ::-1]
		gt_waypoints[:,:, 0] = -1 * gt_waypoints[:,:, 0]

		pred_wp = pred_wp.cpu().numpy()
		pred_wp = pred_wp[:,:, ::-1]
		pred_wp[:,:, 0] = -1 * pred_wp[:,:, 0]
		# pred_wp[:,:, 1] = -1 * pred_wp[:,:, 1]

		front_img = front_img.cpu()
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

		fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
		ax1.imshow(front_img)
		ax1.set_title('image')

		ax2.imshow(front_img)
		ax2.set_title('control_signal')

		trajectories = list(pred_wp) + list(gt_waypoints)
		labels = ["pred", "gt"]
		colors = ["r", "b"]
		ax3 = draw_trajectory_on_ax(ax3, trajectories, labels, colors)

		ax4.imshow(front_img)
		trajectories = [np.concatenate([trajectory, np.zeros((4,1))], 1) for trajectory in trajectories]

		for trajectory_single, label, color in zip(trajectories, labels,colors):
			trajectory_single = trajectory_single[(trajectory_single[..., 0] > 0)]
			if len(trajectory_single) == 0:
				continue

			location = list((p + camera_translation_inv for p in trajectory_single))
			proj_trajectory = np.array(list((camera_intrinsic @ (camera_rotation_matrix_inv @ l) for l in location)))
			proj_trajectory /= proj_trajectory[..., 2:3].repeat(3, -1)

			proj_trajectory = proj_trajectory[(proj_trajectory[..., 0] > 0) & (proj_trajectory[..., 0] < 800)]
			proj_trajectory = proj_trajectory[(proj_trajectory[..., 1] > 0) & (proj_trajectory[..., 1] < 384)]
			ax4.plot(proj_trajectory[:, 0], proj_trajectory[:, 1], 'o-', label=label ,color = color)


		ax4.legend()
		plt.tight_layout()
		output_folder = "vis_aim_v2"
		if not os.path.exists(output_folder):
			os.mkdir(output_folder)
		plt.savefig(os.path.join(output_folder,"%04d.png" % total_number))
		total_number += 1
		plt.close(fig)
