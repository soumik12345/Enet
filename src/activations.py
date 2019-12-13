import torch
from torch.nn import Module
from torch.nn import functional as F



class Mish(Module):
	
	def __init__(self):
		'''Mish Activation Function
		Reference:
			https://arxiv.org/abs/1908.08681
		'''
		super().__init__()

	def forward(self, input):
		return input * torch.tanh(F.softplus(input))