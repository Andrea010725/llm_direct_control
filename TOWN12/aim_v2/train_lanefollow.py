import argparse
import os
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta, Normal

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from model_lanefollow import AIM_V2
from data import CARLA_Data
from pytorch_lightning.plugins import DDPPlugin

from config import GlobalConfig




class AIM_planner(pl.LightningModule):
	def __init__(self, config, lr, device ="cuda"):
		super().__init__()
		self.lr = lr
		self.config = config
		self.model = AIM_V2(config)
		# ckpt = torch.load("/home/wupenghao/transfuser/cilrs/ckpt_epoch_799.pth",map_location='cpu')['model']

		# ckpt = torch.load("/home/wupenghao/transfuser/seg_pretrian/log/semantic_seg_pretrain/epoch=77-last.ckpt",map_location='cpu')['state_dict']

		# new_state_dict = OrderedDict()
		# for key, value in ckpt.items():
		# 	if 'perception.fc' in key:
		# 		continue
		# 	new_key = key.replace("model.perception.","")
		# 	new_state_dict[new_key] = value

		# new_state_dict = OrderedDict()
		# for key, value in ckpt.items():
		# 	if 'encoder.model.fc' in key:
		# 		continue
		# 	new_key = key.replace("encoder.model.","")
		# 	if 'decoder' in new_key:
		# 		continue
		# 	new_state_dict[new_key] = value
		# self.model.perception.load_state_dict(new_state_dict, strict = False)
		self._load_weight()


	def _load_weight(self):
		rl_state_dict = torch.load(self.config.rl_ckpt, map_location='cpu')['policy_state_dict']
		self._load_state_dict(self.model.value_branch, rl_state_dict, 'value_head')
		# self._load_state_dict(self.model.dist_mu, rl_state_dict, 'dist_mu')
		# self._load_state_dict(self.model.dist_sigma, rl_state_dict, 'dist_sigma')

	def _load_state_dict(self, il_net, rl_state_dict, key_word):
		rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
		il_keys = il_net.state_dict().keys()
		assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
		new_state_dict = OrderedDict()
		for k_il, k_rl in zip(il_keys, rl_keys):
			new_state_dict[k_il] = rl_state_dict[k_rl]
		il_net.load_state_dict(new_state_dict)
	
	def forward(self, batch):
		front_img = batch['front_img']
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		state = torch.cat([speed, target_point], 1)

		# inference
		pred = self.model(front_img, state, target_point)
		return pred

	def training_step(self, batch, batch_idx):
		front_img = batch['front_img']
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		state = torch.cat([speed, target_point, command], 1)

		# loss_mask = batch['not_only_ap_brake']

		gt_waypoints = batch['waypoints']
		value = batch['value'].view(-1,1)
		feature = batch['feature']
		# inference
		pred = self.model(front_img, state, target_point)
		
		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight

		value_loss = F.mse_loss(pred['pred_value'], value) * self.config.value_weight
		feature_loss = F.mse_loss(pred['pred_features'], feature) * self.config.features_weight


		wp_loss_lf = F.l1_loss(pred['pred_wp_lf'], gt_waypoints, reduction='none').mean()
		

		loss = wp_loss + speed_loss + (value_loss + feature_loss) + wp_loss_lf
		self.log('train_wp_loss', wp_loss.item())
		self.log('train_wp_loss_lf', wp_loss_lf.item())
		self.log('train_speed_loss', speed_loss.item())
		self.log('train_value_loss', value_loss.item())
		self.log('train_feature_loss', feature_loss.item())
		return loss
		return wp_loss

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
		# optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
		# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=1e-7,
		#                                             factor=0.5,
		#                                             patience=5)
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
		# optimizer = optim.SGD(self.parameters(), lr=LR, momentum=0.9, weight_decay=0.01)
		return [optimizer], [lr_scheduler]
		# return {
		#    'optimizer': optimizer,
		#    'lr_scheduler': lr_scheduler,
		#    'monitor': 'val_wp_loss'
		# }

	def validation_step(self, batch, batch_idx):
		front_img = batch['front_img']
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		state = torch.cat([speed, target_point, command], 1)

		gt_waypoints = batch['waypoints']
		value = batch['value'].view(-1,1)
		feature = batch['feature']
		# inference
		pred = self.model(front_img, state, target_point)
		
		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
		wp_loss_lf = F.l1_loss(pred['pred_wp_lf'], gt_waypoints, reduction='none').mean()
		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
		value_loss = F.mse_loss(pred['pred_value'], value) * self.config.value_weight
		feature_loss = F.mse_loss(pred['pred_features'], feature) * self.config.features_weight


		self.log("val_wp_loss", wp_loss.item(), sync_dist=True)
		self.log("val_wp_loss_lf", wp_loss_lf.item(), sync_dist=True)
		self.log('val_speed_loss', speed_loss.item(), sync_dist=True)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--id', type=str, default='lb2_90routes_rgbhigh_lanefollow_half', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=101, help='Number of train epochs.')
	parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
	parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	# Config
	config = GlobalConfig()

	# Data
	# train_set = CARLA_Data(root=config.train_data, config=config)
	train_set = CARLA_Data(root = '/home/wupenghao/transfuser/data_roach_90/', data_folders =config.train_data, img_aug = config.img_aug)
	val_set = CARLA_Data(root = '/home/wupenghao/transfuser/data_roach_90/', data_folders =config.val_data)
	print(len(train_set))
	print(len(val_set))

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

	aim = AIM_planner(config, args.lr)

	checkpoint_callback = ModelCheckpoint(save_weights_only=False, mode="min", monitor="val_wp_loss", save_top_k=2, save_last=True,
											dirpath=args.logdir, filename="best_{epoch:02d}-{val_wp_loss:.3f}")
	checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
	trainer = pl.Trainer.from_argparse_args(args,
											default_root_dir=args.logdir,
											gpus = 4,
											accelerator='ddp',
											sync_batchnorm=True,
											plugins=DDPPlugin(find_unused_parameters=False),
											profiler='simple',
											benchmark=True,
											log_every_n_steps=1,
											flush_logs_every_n_steps=5,
											callbacks=[checkpoint_callback,
														],
											check_val_every_n_epoch = 1,
											max_epochs = 60
											)

	trainer.fit(aim, dataloader_train, dataloader_val)




		




