import torch
from torch.nn import functional as F
from torch.nn import Module, ReLU, PReLU



class Mish(Module):

	def __init(self):
		'''Mish Activation Function
		Reference:
			https://arxiv.org/abs/1908.08681
		'''
		super().__init__()
	
	def forward(self, input):
		return input * torch.tanh(F.softplus(input))


class Activation(Module):
	
	def __init__(self, name='relu'):
		'''Activation Function
		Params:
			name -> relu/prelu/mish
		'''
		super().__init__()
		self.name = name
		if name == 'relu':
			self.act = ReLU()
		elif name == 'prelu':
			self.act = PReLU()
		elif name == 'mish':
			self.act = Mish()

	def forward(self, input):
		return self.act(input)