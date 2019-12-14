import numpy as np
import torch, cv2, os
from time import time
from tqdm import tqdm
from PIL import Image
from config import CAMVID_CONFIGS
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage


def get_class_weights(loader, num_classes, c=1.02):
    _, labels = next(iter(loader))
    all_labels = labels.flatten()
    each_class = np.bincount(all_labels, minlength=num_classes)
    prospensity_score = each_class / len(all_labels)
    class_weights = 1 / (np.log(c + prospensity_score))
    return class_weights


def read_image(image_file):
    image = Image.open(image_file)
    image = np.array(image)
    image = cv2.resize(image, (512, 512), cv2.INTER_LINEAR)
    image = torch.tensor(image).unsqueeze(0).float()
    image = image.transpose(2, 3).transpose(1, 2).to(device)
    return image


def decode_segmap(image, color_dict):
    label_colours = np.array([
		color_dict['obj0'], color_dict['obj1'],
		color_dict['obj2'], color_dict['obj3'],
		color_dict['obj4'], color_dict['obj5'],
		color_dict['obj6'], color_dict['obj7'],
		color_dict['obj8'], color_dict['obj9'],
		color_dict['obj10'], color_dict['obj11']
	]).astype(np.uint8)
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, 12):
        r[image == l] = label_colours[l, 0]
        g[image == l] = label_colours[l, 1]
        b[image == l] = label_colours[l, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = b
    rgb[:, :, 1] = g
    rgb[:, :, 2] = r
    return rgb


def predict_rgb(model, tensor, color_dict):
    with torch.no_grad():
        out = model(tensor.float()).squeeze(0)
    out = out.data.max(0)[1].cpu().numpy()
    return decode_segmap(out, color_dict)



def visualize_batch(loader, model, color_dict, n_images):
	'''Visualize batch from model
	Params:
		loader		-> Data loader
		model		-> Model for prediction
		color_dict	-> Class color dict
		n_images	-> Number of images (< batch size)
	'''
	x_batch, y_batch = next(iter(loader))
	fig, axes = plt.subplots(nrows = n_images, ncols = 3, figsize = (16, 16))
	plt.setp(axes.flat, xticks = [], yticks = [])
	c = 1
	for i, ax in enumerate(axes.flat):
		if i % 3 == 0:
			ax.imshow(ToPILImage()(x_batch[c]))
			ax.set_xlabel('Image_' + str(c))
		elif i % 3 == 1:
			ax.imshow(
				decode_segmap(
					y_batch[c],
					color_dict
				)
			)
			ax.set_xlabel('Ground_Truth_' + str(c))
		elif i % 3 == 2:
			ax.imshow(
				predict_rgb(
					enet,
					x_batch[c].unsqueeze(0).to(device),
					color_dict
				)
			)
			ax.set_xlabel('Predicted_Mask_' + str(c))
			c += 1
	plt.show()


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
		x = Image.open(image_id)
		x = np.array(x)
		x = [
			cv2.resize(
				x, (self.height, self.width),
				cv2.INTER_LINEAR
			)
		]
		x = np.stack(x, axis=2)
		x = torch.tensor(x).transpose(0, 2).transpose(1, 3)
		# Read Mask
		y = Image.open(label_id)
		y = np.array(y)
		y = [cv2.resize(
			y, (self.height, self.width),
			cv2.INTER_NEAREST
		)]
		y = torch.tensor(y)
		return x.squeeze(), y.squeeze()



def train(
	model, train_dataloader, val_dataloader,
	device, criterion, optimizer, train_step_size, val_step_size,
	visualize_every, save_every, save_location, save_prefix, epochs):
	'''Training Function for Campvid
	Params:
		model				-> Model
		train_dataloader	-> Train Data Loader
		val_dataloader		-> Validation Data Loader
		device				-> Training Device
		criterion           -> Loss Function
		optimizer			-> Optimizer
		train_step_size		-> Training Step Size
		val_step_size		-> Validation Step Size
		visualize_every		-> Visualization Checkpoint
		save_every			-> Saving Checkpoint
		save_location		-> Checkpoint Saving Location
		save_prefix			-> Checkpoint Prefix
		epochs				-> Number of Training epochs
	'''
	try:
		os.mkdir(save_location)
	except:
		pass
	train_loss_history, val_loss_history = [], []
	train_time = []
	for epoch in range(1, epochs + 1):
		print('Epoch {}\n'.format(epoch))
		# Training
		start = time()
		train_loss = 0
		model.train()
		for step in tqdm(range(train_step_size)):
			x_batch, y_batch = next(iter(train_dataloader))
			x_batch = x_batch.squeeze().to(device)
			y_batch = y_batch.squeeze().to(device)
			optimizer.zero_grad()
			out = model(x_batch.float())
			loss = criterion(out, y_batch.long())
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
		train_loss_history.append(train_loss / train_step_size)
		print('\nTraining Loss: {}'.format(train_loss_history[-1]))
		train_time.append(time() - start)
		print('Training Time: {} seconds'.format(train_time[-1]))
		# Validation
		val_loss = 0
		model.eval()
		for step in tqdm(range(val_step_size)):
			x_val, y_val = next(iter(val_dataloader))
			x_val = x_val.squeeze().to(device)
			y_val = y_val.squeeze().to(device)
			out = model(x_val.float())
			out = out.data.max(1)[1]
			val_loss += (y_val.long() - out.long()).float().mean()
		val_loss_history.append(val_loss)
		print('\nValidation Loss: {}'.format(val_loss))
		# Visualization
		if epoch % visualize_every == 0:
			visualize_batch(val_dataloader, model, CAMVID_CONFIGS['class_colors'], 1)
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
			print('Checkpoint saved')
	print(
        '\nTraining Done.\nTraining Mean Loss: {:6f}\nValidation Mean Loss: {:6f}'.format(
            sum(train_loss_history) / epochs,
            sum(val_loss_history) / epochs
        )
    )
	return train_loss_history, val_loss_history, train_time