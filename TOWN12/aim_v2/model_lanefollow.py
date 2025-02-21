from collections import deque

import numpy as np
from numpy.testing._private.utils import measure
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models


class ImageCNN(nn.Module):
	""" Encoder network for image input list.
	Args:
		c_dim (int): output dimension of the latent embedding
		normalize (bool): whether the input images should be normalized
	"""

	def __init__(self):
		super().__init__()
		self.features = models.resnet34(pretrained=True)
		# self.features.fc = nn.Sequential()

	def forward(self, inputs):

		c = self.features(inputs)
		return c

def normalize_imagenet(x):
	""" Normalize input images according to ImageNet standards.
	Args:
		x (tensor): input images
	"""
	x = x.clone()
	x[:, 0] = (x[:, 0] - 0.485) / 0.229
	x[:, 1] = (x[:, 1] - 0.456) / 0.224
	x[:, 2] = (x[:, 2] - 0.406) / 0.225
	return x


class PIDController(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative


class AIM_V2(nn.Module):

	def __init__(self, config):
		super().__init__()
		self.pred_len = config.pred_len
		self.config = config

		self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
		self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

		self.perception = models.resnet34(pretrained=True)
		# self.perception.fc = nn.Sequential()

		self.measurements = nn.Sequential(
							nn.Linear(1+2+6, 128),
							nn.ReLU(inplace=True),
							nn.Linear(128, 128),
							nn.ReLU(inplace=True),
						)

		self.join = nn.Sequential(
							nn.Linear(128+1000, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)
		self.decoder = nn.GRUCell(input_size=4, hidden_size=256)
		self.output = nn.Linear(256, 2)

		# lanefollow

		self.measurements_lf = nn.Sequential(
							nn.Linear(1, 64),
							nn.ReLU(inplace=True),
							nn.Linear(64, 64),
							nn.ReLU(inplace=True),
						)

		self.join_lf = nn.Sequential(
							nn.Linear(64+1000, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)
		self.decoder_lf = nn.GRUCell(input_size=2, hidden_size=256)
		self.output_lf = nn.Linear(256, 2)



		self.speed_branch = nn.Sequential(
							nn.Linear(1000, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							nn.Dropout2d(p=0.5),
							nn.ReLU(inplace=True),
							nn.Linear(256, 1),
						)

		self.value_branch = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)

	def forward(self, img, state, target_point):
		feature_emb = self.perception(img)
		measurement_feature = self.measurements(state)
		j = self.join(torch.cat([feature_emb, measurement_feature], 1))

		measurement_feature_lf = self.measurements_lf(state[:,0:1])
		j_lf = self.join_lf(torch.cat([feature_emb, measurement_feature_lf], 1))

		outputs = {'pred_speed': self.speed_branch(feature_emb)}
		outputs['pred_value'] = self.value_branch(j)
		outputs['pred_features'] = j


		z_lf = j_lf
		output_wp_lf = list()

		# initial input variable to GRU
		x_lf = torch.zeros(size=(z_lf.shape[0], 2), dtype=z_lf.dtype).to(z_lf.device)

		# autoregressive generation of output waypoints
		for _ in range(self.pred_len):
			x_in_lf = x_lf
			z_lf = self.decoder_lf(x_in_lf, z_lf)
			dx_lf = self.output_lf(z_lf)
			x_lf = dx_lf + x_lf
			output_wp_lf.append(x_lf)

		pred_wp_lf = torch.stack(output_wp_lf, dim=1)
		outputs['pred_wp_lf'] = pred_wp_lf



		
		z = j
		output_wp = list()

		# initial input variable to GRU
		x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).to(z.device)

		# autoregressive generation of output waypoints
		for _ in range(self.pred_len):
			x_in = torch.cat([x, target_point], dim=1)
			z = self.decoder(x_in, z)
			dx = self.output(z)
			x = dx + x
			output_wp.append(x)

		pred_wp = torch.stack(output_wp, dim=1)
		outputs['pred_wp'] = pred_wp
		return outputs

	def control_pid(self, waypoints, velocity, target = None):
		''' 
		Predicts vehicle control with a PID controller.
		Args:
			waypoints (tensor): predicted waypoints
			velocity (tensor): speedometer input
		'''
		assert(waypoints.size(0)==1)
		waypoints = waypoints[0].data.cpu().numpy()

		# waypoints = waypoints[:,::-1]
		waypoints[:,1] *= -1
		speed = velocity[0].data.cpu().numpy()

		aim = (waypoints[1] + waypoints[0]) / 2.0
		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		steer = self.turn_controller.step(angle)
		steer = np.clip(steer, -1.0, 1.0)

		desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
		brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

		delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.config.max_throttle)
		throttle = throttle if not brake else 0.0
		if target is not None:
			metadata = {
				'speed': float(speed.astype(np.float64)),
				'steer': float(steer),
				'throttle': float(throttle),
				'brake': float(brake),
				'wp_4': tuple(waypoints[3].astype(np.float64)),
				'wp_3': tuple(waypoints[2].astype(np.float64)),
				'wp_2': tuple(waypoints[1].astype(np.float64)),
				'wp_1': tuple(waypoints[0].astype(np.float64)),
				'desired_speed': float(desired_speed.astype(np.float64)),
				'angle': float(angle.astype(np.float64)),
				'aim': tuple(aim.astype(np.float64)),
				'delta': float(delta.astype(np.float64)),
				'target': tuple(target[0].data.cpu().numpy().astype(np.float64)),
			}
		else:
			metadata = {
				'speed': float(speed.astype(np.float64)),
				'steer': float(steer),
				'throttle': float(throttle),
				'brake': float(brake),
				'wp_4': tuple(waypoints[3].astype(np.float64)),
				'wp_3': tuple(waypoints[2].astype(np.float64)),
				'wp_2': tuple(waypoints[1].astype(np.float64)),
				'wp_1': tuple(waypoints[0].astype(np.float64)),
				'desired_speed': float(desired_speed.astype(np.float64)),
				'angle': float(angle.astype(np.float64)),
				'aim': tuple(aim.astype(np.float64)),
				'delta': float(delta.astype(np.float64)),
			}
		return steer, throttle, brake, metadata


	def control_pid_neat(self, waypoints, velocity, target):
		''' Predicts vehicle control with a PID controller.
		Args:
			waypoints (tensor): output of self.plan()
			velocity (tensor): speedometer input
		'''
		assert(waypoints.size(0)==1)
		waypoints = waypoints[0].data.cpu().numpy()
		target = target.squeeze().data.cpu().numpy()

		# flip y (forward is negative in our waypoints)
		waypoints[:,1] *= -1
		target[1] *= -1

		# iterate over vectors between predicted waypoints
		num_pairs = len(waypoints) - 1
		best_norm = 1e5
		desired_speed = 0
		aim = waypoints[0]
		for i in range(num_pairs):
			# magnitude of vectors, used for speed
			desired_speed += np.linalg.norm(
					waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

			# norm of vector midpoints, used for steering
			norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
			if abs(self.config.aim_dist-best_norm) > abs(self.config.aim_dist-norm):
				aim = waypoints[i]
				best_norm = norm

		aim_last = waypoints[-1] - waypoints[-2]

		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
		angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		# choice of point to aim for steering, removing outlier predictions
		# use target point if it has a smaller angle or if error is large
		# predicted point otherwise
		# (reduces noise in eg. straight roads, helps with sudden turn commands)
		use_target_to_aim = np.abs(angle_target) < np.abs(angle)
		use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
		if use_target_to_aim:
			angle_final = angle_target
		else:
			angle_final = angle

		steer = self.turn_controller.step(angle_final)
		steer = np.clip(steer, -1.0, 1.0)

		speed = velocity[0].data.cpu().numpy()
		brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

		delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
		throttle = self.speed_controller.step(delta)
		throttle = np.clip(throttle, 0.0, self.config.max_throttle)
		throttle = throttle if not brake else 0.0

		metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_4': tuple(waypoints[3].astype(np.float64)),
			'wp_3': tuple(waypoints[2].astype(np.float64)),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'angle_last': float(angle_last.astype(np.float64)),
			'angle_target': float(angle_target.astype(np.float64)),
			'angle_final': float(angle_final.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}

		return steer, throttle, brake, metadata