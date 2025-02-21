from collections import deque
import numpy as np
import torch 
from torch import nn
from resnet import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class LCD(nn.Module):

	def __init__(self):
		super().__init__()
		self.perception = resnet34(pretrained=True)
		# self.perception.fc = nn.Sequential()
		self.compress_traj_feat = nn.Sequential(
							nn.Linear(1000, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)

		self.cls_branch = nn.Sequential(
							nn.Linear(1000, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							nn.Dropout2d(p=0.5),
							nn.ReLU(inplace=True),
							nn.Linear(256, 3),
						)

		self.decoder_traj = nn.GRUCell(input_size=2, hidden_size=256)
		self.output_traj = nn.Linear(256, 2)

		

	def forward(self, img):
		feature_emb, _ = self.perception(img)
		outputs = {}
		outputs['pred_cls'] = self.cls_branch(feature_emb)

		mid_feat = self.compress_traj_feat(feature_emb)
		# outputs['pred_features_traj'] = mid_feat

		z = mid_feat
		output_wp = list()
		traj_hidden_state = list()


		# initial input variable to GRU
		x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)
		hard_code = 30

		# autoregressive generation of output waypoints
		for _ in range(hard_code):
			x_in = x
			z = self.decoder_traj(x_in, z)
			traj_hidden_state.append(z)
			dx = self.output_traj(z)
			x = dx + x
			output_wp.append(x)

		pred_wp = torch.stack(output_wp, dim=1)
		outputs['pred_wp'] = pred_wp
		return outputs


if __name__ == '__main__':
	img = torch.rand(2, 3, 384, 800)
	model = LCD()
	out = model(img)
	print(out)