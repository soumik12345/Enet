import torch
from torch.nn import (
	Module, Conv2d, ReLU, PReLU,
	Upsample, MaxPool2d, Sequential, MaxUnpool2d,
	BatchNorm2d, AdaptiveAvgPool2d, ConvTranspose2d
)
from torchvision.models import resnet50


class InitialBlock(Module):
	
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



class RegularBottleneck(Module):

	def __init__(
		self, channels, internal_ratio=4, kernel_size=3, padding=0,
		dilation=1, asymmetric=False, dropout_prob=0, bias=False, relu=True):
		'''Enet Regular Bottleneck Block
		Reference: https://arxiv.org/abs/1606.02147
		Params:
			channels		-> Number of input and output channels
			internal_ratio	-> Scale factor for channels
			kernel_size		-> Kernel size for conv layer, block 2, main branch
			padding			-> Zero padding for input
			dilation		-> Dilation for conv layer, block 2, main branch
			assymetric		-> conv layer, block 2, main branch is assymetric if true
			dropout_prob	-> Probability for dropout
			bias			-> Use a bias or not
			relu			-> Use ReLU activation if true
		'''
		
		super().__init__()
		internal_channels = channels // internal_ratio
		
		### Main Brach ###
		
		# Block 1 Conv 1x1
		self.main_conv_block_1 = Sequential(
			Conv2d(
				channels, internal_channels,
				kernel_size=1, stride=1, bias=bias
			),
			BatchNorm2d(internal_channels),
			ReLU() if relu else PReLU()
		)

		# Block 2
		if assymetric:
			self.main_conv_block_2 = Sequential(
				Conv2d(
					internal_channels, internal_channels,
					kernel_size=(kernel_size, 1), stride=1,
					padding=(padding, 0), dilation=dilation, bias=bias
				),
				BatchNorm2d(internal_channels),
				ReLU() if relu else PReLU(),
				Conv2d(
					internal_channels, internal_channels,
					kernel_size=(1, kernel_size), stride=1,
					padding=(0, padding), dilation=dilation, bias=bias
				),
				BatchNorm2d(internal_channels),
				ReLU() if relu else PReLU(),
			)
		else:
			self.main_conv_block_2 = Sequential(
				Conv2d(
					internal_channels, internal_channels,
					kernel_size=kernel_size, stride=1,
					padding=padding, dilation=dilation, bias=bias
				),
				BatchNorm2d(internal_channels),
				ReLU() if relu else PReLU(),
			)
		
		# Block 3 Conv 1x1
		self.main_conv_block_3 = Sequential(
			Conv2d(
				internal_channels, channels,
				kernel_size=1, stride=1, bias=bias
			),
			BatchNorm2d(channels),
			ReLU() if relu else PReLU(),
		)

		# Dropout Regularization
		self.dropout = Dropout2d(p=dropout_prob)

		# Activation
		self.activation = ReLU() if relu else PReLU()
	

	def forward(self, x):
		'''Forward Pass for RegularBottleneck'''
		secondary_brach = x
		main_brach = self.main_conv_block_1(x)
		main_brach = self.main_conv_block_2(main_brach)
		main_brach = self.main_conv_block_3(main_brach)
		main_brach = self.dropout(main_brach)
		output = main_brach + secondary_brach
		output = self.activation(output)
		return output



class DownsampleBottleneckBlock(Module):

	def __init__(
		self, in_channels, out_channels, internal_ratio=4,
		return_indices=False, dropout_prob=0, bias=False, relu=True):
		'''Enet DownSampling Bottleneck Block
		Reference: https://arxiv.org/abs/1606.02147
		Params:
			in_channels		-> Number of input channels
			out_channels	-> Number of output channels
			internal_ratio	-> Scale factor for channels
			return_indices	-> Returns max indices if true
			dropout_prob	-> Probability for dropout
			bias			-> Use a bias or not
			relu			-> Use ReLU activation if true
		'''
		super().__init__()
		internal_channels = in_channels // internal_ratio
		self.return_indices = return_indices

		### Main Branch ###

		# Block 1 Conv 1x1
		self.main_conv_block_1 = Sequential(
			Conv2d(
				in_channels, internal_channels,
				kernel_size=2, stride=2, bias=bias
			),
			BatchNorm2d(internal_channels),
			ReLU() if relu else PReLU()
		)

		# Block 2 Conv 3x3
		self.main_conv_block_2 = Sequential(
			Conv2d(
				internal_channels, internal_channels,
				kernel_size=3, stride=1, padding=1, bias=bias
			),
			BatchNorm2d(internal_channels),
			ReLU() if relu else PReLU()
		)

		# Block 2 Conv 1x1
		self.main_conv_block_3 = Sequential(
			Conv2d(
				internal_channels, out_channels,
				kernel_size=1, stride=1, bias=bias
			),
			BatchNorm2d(out_channels),
			ReLU() if relu else PReLU()
		)

		### Secondary Branch ###
		self.secondary_maxpool = MaxPool2d(
			2, stride=2,
			return_indices=return_indices
		)

		# Dropout Regularization
		self.dropout = Dropout2d(p=dropout_prob)

		# Activation
		self.activation = ReLU() if relu else PReLU()
	

	def forward(self, x):
		'''Forward Pass for DownsampleBottleneckBlock'''
		# Main Branch
		main_brach = self.main_conv_block_1(x)
		main_brach = self.main_conv_block_2(main_brach)
		main_brach = self.main_conv_block_3(main_brach)
		# Secondary Branch
		if self.return_indices:
			secondary_brach, max_indices = self.secondary_maxpool(x)
		else:
			secondary_branch = self.secondary_maxpool(x)
		# Padding
		n, ch_main, h, w = main_brach.size()
		ch_sec = secondary_branch.size()[1]
		padding = torch.zeros(n, ch_main - ch_sec, h, w)
		if secondary_branch.is_cuda:
			padding = padding.cuda()
		# Concatenate
		secondary_branch = torch.cat((secondary_branch, padding), 1)
		output = secondary_branch + primary_branch
		output = self.activation(output)
		if self.return_indices:
			return output, max_indices
		else:
			return output



class UpsampleBottleneckBlock(Module):

	def __init__(
		self, in_channels, out_channels,
		internal_ratio=4, dropout_prob=0,
		bias=False, relu=True):
		'''Enet Upsampling Bottleneck Block
		Reference: https://arxiv.org/abs/1606.02147
		Params:
			in_channels		-> Number of input channels
			out_channels	-> Number of output channels
			internal_ratio	-> Scale factor for channels
			dropout_prob	-> Probability for dropout
			bias			-> Use a bias or not
			relu			-> Use ReLU activation if true
		'''
		super().__init__()

		### Main Branch ###

		# Block 1 Conv 1x1
		self.main_branch_conv_1 = Sequential(
			Conv2d(
				in_channels, internal_channels,
				kernel_size=1, bias=bias
			),
			BatchNorm2d(internal_channels),
			ReLU() if relu else PReLU()
		)

		# Block 2 Transposed Convolution
		self.main_branch_transpose_conv_2 = ConvTranspose2d(
			internal_channels, internal_channels,
			kernel_size=2, stride=2, bias=bias
		)
		self.main_branch_bn_2 = BatchNorm2d(internal_channels)
		self.main_branch_act_2 = ReLU() if relu else PReLU

		# Block 3 Conv 1x1
		self.main_branch_conv_3 = Sequential(
			Conv2d(
				internal_channels, out_channels,
				kernel_size=1, bias=bias
			),
			BatchNorm2d(out_channels),
			ReLU() if relu else PReLU()
		)

		### Secondary Branch ###
		self.secondary_conv = Sequential(
			Conv2d(
				in_channels, out_channels,
				kernel_size=1, bias=bias
			),
			BatchNorm2d(out_channels)
		)
		self.secondary_unpool = MaxUnpool2d(kernel_size=2)

		# Dropout Regularization
		self.dropout = Dropout2d(p=dropout_prob)

		# Activation
		self.activation = ReLU() if relu else PReLU()
	

	def forward(self, x, max_indices, output_size):
		'''Forward Pass for UpsampleBottleneckBlock'''
		# Main Branch
		main_branch = self.main_branch_conv_1(x)
		main_branch = self.main_branch_transpose_conv_2(main_branch, output_size=output_size)
		main_branch = self.main_branch_bn_2(main_branch)
		main_branch = self.main_branch_act_2(main_branch)
		main_branch = self.main_branch_conv_3(main_branch)
		main_branch = self.dropout(main_branch)
		# Secondary Branch
		secondary branch = self.secondary_conv(x)
		secondary_brach = self.secondary_unpool(
			secondary_brach, max_indices,
			output_size=output_size
		)
		# Concatenate
		output = main_branch + secondary_brach
		output = self.activation(output)
		return output