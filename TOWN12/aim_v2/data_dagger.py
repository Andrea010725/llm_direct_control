import os
import json
from PIL import Image
import random

import numpy as np
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

from augment import hard as augmenter

class CARLA_Data_Dagger(Dataset):

	def __init__(self, root, root_dagger, img_aug = False):
		
		self.img_aug = img_aug
		self.front_img = []
		self.x = []
		self.y = []
		self.x_command = []
		self.y_command = []
		self.command = []
		self.theta = []
		self.speed = []

		self.future_x = []
		self.future_y = []
		self.future_theta = []

		self.value = []
		self.feature = []

		self.target_command = []
		self.target_gps = []

		self._batch_read_number = 0

		init_num = 0

		for sub_root in root:
			data_both = np.load(os.path.join(sub_root, "packed_data_both.npy"), allow_pickle=True).item()
			init_num += len(data_both['front_img'])

		print(init_num)
		total_num = init_num
	
		for sub_root in root_dagger:
			data = np.load(os.path.join(sub_root, "packed_data.npy"), allow_pickle=True).item()
			total_num += len(data['front'])
		print(total_num - init_num)

		random.seed(1)

		for sub_root in root_dagger:
			data = np.load(os.path.join(sub_root, "packed_data.npy"), allow_pickle=True).item()

			current_num = len(data['front'])
			shuffle_index = list(range(current_num))
			shuffle_total_number = min(current_num, int((init_num * current_num / total_num)*1.2))
			shuffle_index = random.sample(shuffle_index, shuffle_total_number)

			self.front_img += [data['front'][_] for _ in shuffle_index]
			self.x += [data['x'][_] for _ in shuffle_index]
			self.y += [data['y'][_] for _ in shuffle_index]
			self.theta += [data['theta'][_] for _ in shuffle_index]

			# self.target_command += data['target_command']
			# self.target_gps += data['target_gps']

			self.x_command += [data['x_command_far'][_] for _ in shuffle_index]
			self.y_command += [data['y_command_far'][_] for _ in shuffle_index]
			self.command += [data['command_far'][_] for _ in shuffle_index]
			self.speed += [data['speed'][_] for _ in shuffle_index]

			self.future_x += [data['future_x'][_] for _ in shuffle_index]
			self.future_y += [data['future_y'][_] for _ in shuffle_index]
			self.future_theta += [data['future_theta'][_] for _ in shuffle_index]

			self.value += [data['value'][_] for _ in shuffle_index]
			self.feature += [data['feature'][_] for _ in shuffle_index]

		for sub_root in root:
			data_traj = np.load(os.path.join(sub_root, "packed_data_traj.npy"), allow_pickle=True).item()
			data_both = np.load(os.path.join(sub_root, "packed_data_both.npy"), allow_pickle=True).item()

			current_num = len(data_both['front_img'])
			shuffle_index = list(range(current_num))
			shuffle_total_number = min(current_num , int((init_num * current_num / total_num)*1.2))
			shuffle_index = random.sample(shuffle_index, shuffle_total_number)

			self.front_img += [data_both['front_img'][_] for _ in shuffle_index]
			self.x += [data_both['x'][_] for _ in shuffle_index]
			self.y += [data_both['y'][_] for _ in shuffle_index]
			self.theta += [data_both['theta'][_] for _ in shuffle_index]

			# self.target_command += data['target_command']
			# self.target_gps += data['target_gps']

			self.x_command += [data_traj['x_command_far'][_] for _ in shuffle_index]
			self.y_command += [data_traj['y_command_far'][_] for _ in shuffle_index]
			self.command += [data_traj['command_far'][_] for _ in shuffle_index]
			self.speed += [data_both['speed'][_] for _ in shuffle_index]

			self.future_x += [data_traj['future_x'][_] for _ in shuffle_index]
			self.future_y += [data_traj['future_y'][_] for _ in shuffle_index]
			self.future_theta += [data_traj['future_theta'][_] for _ in shuffle_index]

			self.value += [data_both['value'][_] for _ in shuffle_index]
			self.feature += [data_both['feature'][_] for _ in shuffle_index]
		self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

		print(len(self.front_img))

	def __len__(self):
		"""Returns the length of the dataset. """
		return len(self.front_img)

	def __getitem__(self, index):
		"""Returns the item at index idx. """
		data = dict()
		data['front_img'] = self.front_img[index]

		if self.img_aug:
			data['front_img'] = self._im_transform(augmenter(self._batch_read_number).augment_image(np.array(
					Image.open(self.front_img[index]))))
		else:
			data['front_img'] = self._im_transform(np.array(
					Image.open(self.front_img[index])))

		# fix for theta=nan in some measurements
		if np.isnan(self.theta[index]):
			self.theta[index] = 0.

		ego_x = self.x[index]
		ego_y = self.y[index]
		ego_theta = self.theta[index] 




		waypoints = []
		for i in range(4):
			# waypoint is the transformed version of the origin in local coordinates
			# we use 90-theta instead of theta
			# LBC code uses 90+theta, but x is to the right and y is downwards here
			local_waypoint = transform_2d_points(np.zeros((1,3)), 
				np.pi/2-self.future_theta[index][i], -self.future_y[index][i], -self.future_x[index][i], np.pi/2-ego_theta, -ego_y, -ego_x)
			waypoints.append(tuple(local_waypoint[0,:2]))

		data['waypoints'] = np.array(waypoints)

		# convert x_command, y_command to local coordinates
		# taken from LBC code (uses 90+theta instead of theta)
		R = np.array([
			[np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
			[np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
			])
		local_command_point = np.array([self.y_command[index]-ego_y, self.x_command[index]-ego_x])
		local_command_point = R.T.dot(local_command_point)
		data['target_point'] = local_command_point[:2]

		# convert x_command, y_command to local coordinates
		# local_command_point = np.array([self.target_gps[index][0], self.target_gps[index][1]])*np.array([111324.60662786, 111319.490945])
		# local_command_point = np.array([(local_command_point[0]-ego_y), (local_command_point[1]-ego_x)])
		# local_command_point = R.T.dot(local_command_point)
		# data['target_point'] = local_command_point[:2]


		# waypoint processing to local coordinates
		# waypoints = []
		# for i in range(4):
		# 	future_x = self.future_x[index][i]
		# 	future_y = self.future_y[index][i]
		# 	# x forward y right
		# 	target_vec_in_global = np.array([future_x - ego_x, -1*(future_y - ego_y), 0])
		# 	local_waypoint = vec_global_to_ref(target_vec_in_global, {'yaw':np.rad2deg(ego_theta)-90, 'pitch':0, 'roll':0})
		# 	if local_waypoint[0] < 0:
		# 		if abs(local_waypoint[0]) < 0.01:
		# 			local_waypoint[0] = 0
		# 	waypoints.append([local_waypoint[0], local_waypoint[1]])

		# data['waypoints'] = np.array(waypoints)

		# # convert x_command, y_command to local coordinates
		# local_command_point = np.array([self.x_command[index]-ego_x, -1*(self.y_command[index]-ego_y), 0])
		# local_command_point = vec_global_to_ref(local_command_point, {'yaw':np.rad2deg(ego_theta)-90, 'pitch':0, 'roll':0})
		# data['target_point'] = local_command_point[:2]




		# VOID = -1
		# LEFT = 1
		# RIGHT = 2
		# STRAIGHT = 3
		# LANEFOLLOW = 4
		# CHANGELANELEFT = 5
		# CHANGELANERIGHT = 6
		command = self.command[index]
		# command = self.target_command[index]
		if command < 0:
			command = 4
		command -= 1
		assert command in [0, 1, 2, 3, 4, 5]
		cmd_one_hot = [0] * 6
		cmd_one_hot[command] = 1
		data['target_command'] = torch.tensor(cmd_one_hot)



		data['speed'] = self.speed[index]
		data['feature'] = self.feature[index]
		data['value'] = self.value[index]
		self._batch_read_number += 1
		return data


def scale_and_crop_image(image, scale=1, crop=256):
	"""
	Scale and crop a PIL image, returning a channels-first numpy array.
	"""
	(width, height) = (int(image.width // scale), int(image.height // scale))
	im_resized = image.resize((width, height))
	image = np.asarray(im_resized)
	start_x = height//2 - crop//2
	start_y = width//2 - crop//2
	cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
	cropped_image = np.transpose(cropped_image, (2,0,1))
	return cropped_image

def transform_2d_points(xyz, r1, t1_x, t1_y, r2, t2_x, t2_y):
	"""
	Build a rotation matrix and take the dot product.
	"""
	# z value to 1 for rotation
	xy1 = xyz.copy()
	xy1[:,2] = 1

	c, s = np.cos(r1), np.sin(r1)
	r1_to_world = np.matrix([[c, s, t1_x], [-s, c, t1_y], [0, 0, 1]])

	# np.dot converts to a matrix, so we explicitly change it back to an array
	world = np.asarray(r1_to_world @ xy1.T)

	c, s = np.cos(r2), np.sin(r2)
	r2_to_world = np.matrix([[c, s, t2_x], [-s, c, t2_y], [0, 0, 1]])
	world_to_r2 = np.linalg.inv(r2_to_world)

	out = np.asarray(world_to_r2 @ world).T
	
	# reset z-coordinate
	out[:,2] = xyz[:,2]

	return out

def rot_to_mat(roll, pitch, yaw):
	roll = np.deg2rad(roll)
	pitch = np.deg2rad(pitch)
	yaw = np.deg2rad(yaw)

	yaw_matrix = np.array([
		[np.cos(yaw), -np.sin(yaw), 0],
		[np.sin(yaw), np.cos(yaw), 0],
		[0, 0, 1]
	])
	pitch_matrix = np.array([
		[np.cos(pitch), 0, -np.sin(pitch)],
		[0, 1, 0],
		[np.sin(pitch), 0, np.cos(pitch)]
	])
	roll_matrix = np.array([
		[1, 0, 0],
		[0, np.cos(roll), np.sin(roll)],
		[0, -np.sin(roll), np.cos(roll)]
	])

	rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
	return rotation_matrix


def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
	R = rot_to_mat(ref_rot_in_global['roll'], ref_rot_in_global['pitch'], ref_rot_in_global['yaw'])
	np_vec_in_global = np.array([[target_vec_in_global[0]],
								 [target_vec_in_global[1]],
								 [target_vec_in_global[2]]])
	np_vec_in_ref = R.T.dot(np_vec_in_global)
	return np_vec_in_ref[:,0]


# data_root = "data/roach_detector_data_collect"

# root = ["town01_short", "town01_tiny", "town02_short"]

# root = [os.path.join(data_root, _) for _ in root]
# from config import GlobalConfig
# cfg = GlobalConfig()
# data_set = CARLA_Data(cfg.train_data, cfg.train_data_dagger1)
# dataloader_train = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=1)

# for i, batch in enumerate(dataloader_train):
# 	x = batch