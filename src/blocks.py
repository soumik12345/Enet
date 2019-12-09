import torch
from torch.nn import (
	Upsample, MaxPool2d,
	Module, Conv2d, ReLU, PReLU,
	BatchNorm2d, AdaptiveAvgPool2d
)
from torchvision.models import resnet50


class InitialBlock(nn.Module):
	
	def __init__(self, in_channels, out_channels, bias=False, relu=True):
		'''Enet Initial Block
		Reference: https://arxiv.org/abs/1606.02147
		Params:
			in_channels  -> Number of input channels
			out_channels -> Number of output channels
			bias		 -> Use bias in the convolution layer
			relu		 -> Use relu activation or not
		'''
		super().__init__()
		self.main_branch = Conv2d(
			in_channels, out_channels - 3,
			kernel_size=3, stride=2,
			padding=1, bias=bias
		)
		self.secondary_branch = MaxPool2d(3, stride=2, padding=1)
		self.batch_norm = BatchNorm2d(out_channels)
		self.activation = ReLU() if relu else PReLU()

	def forward(self, x):
		'''InitialBlock Forward Pass'''
		main = self.main_branch(x)
		secondary = self.secondary_branch(x)
		output = torch.cat((main, secondary), 1)
		output = self.batch_norm(out)
		output = self.activation(output)
		return output