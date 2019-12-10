import torch, cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class CamVidDataset(Dataset):

	def __init__(self, images, labels, height, width):
		'''Camvid Dataset
		Params:
			images -> List of image files
			labels -> List of label files
			height -> Height of image and label
			width  -> Width of image and label
		'''
		self.images = images
		self.labels = labels
		self.height = height
		self.width = width
	
	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, index):
		image_id = self.images[index]
		label_id = self.labels[index]
		# Read Image
		x = plt.imread(image_id)
		x = [cv2.resize(x, (self.height, self.width), cv2.INTER_NEAREST)]
		x = np.stack(x, axis=2)
		x = torch.tensor(x).transpose(0, 2).transpose(1, 3)
		# Read Mask
		y = Image.open(label_id)
		y = np.array(y)
		y = cv2.resize(y, (self.height, self.width), cv2.INTER_NEAREST)
		y = torch.tensor([y])
		return x, y



def train(
	model, train_dataloader, val_dataloader,
	device, loss, optimizer, train_step_size, val_step_size,
	save_every, save_location, save_prefix, epochs):
	'''Training Function for Campvid
	Params:
		model				-> Model
		train_dataloader	-> Train Data Loader
		val_dataloader		-> Validation Data Loader
		device				-> Training Device
		loss				-> Loss Function
		optimizer			-> Optimizer
		train_step_size		-> Training Step Size
		val_step_size		-> Validation Step Size
		save_every			-> Saving Checkpoint
		save_location		-> Checkpoint Saving Location
		save_prefix			-> Checkpoint Prefix
		epochs				-> Number of Training epochs
	'''
	train_loss_history, val_loss_history = [], []
	for epoch in range(1, epochs + 1):
		print('Epoch {}\n'.format(epoch))
		# Training
		train_loss = 0
		model.train()
		for step in tqdm(range(train_step_size)):
			x_batch, y_batch = next(iter(train_dataloader))
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)
			optimizer.zero_grad()
			out = model(x_batch.float())
			loss = criterion(out, y_batch.long())
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		train_loss_history.append(train_loss)
		print('\nTraining Loss: {}'.format(train_loss))
		# Validation
		val_loss = 0
		model.eval()
		for step in tqdm(range(val_step_size)):
			x_val, y_val = next(iter(val_dataloader))
			x_val = x_val.to(device)
			y_val = y_val.to(device)
			out = model(x_val.float())
			out = out.data.max(1)[1]
			val_loss += (y_val.long() - out.long()).sum()
		val_loss_history.append(val_loss)
		print('\nValidation Loss: {}'.format(val_loss))
		# Checkpoints
		if epoch % save_every == 0:
			checkpoint = {
				'epoch' : epoch,
				'train_loss' : train_loss,
				'val_loss' : val_loss,
				'state_dict' : model.state_dict()
			}
			torch.save(
				checkpoint,
				'{}/{}-{}-{}-{}.pth'.format(
					save_location, save_prefix,
					epoch, train_loss, val_loss
				)
			)
	print('\nTraining Done.\nTotal Mean Loss: {:6f}'.format(sum(train_losses) / epochs))
	return train_loss_history, val_loss_history