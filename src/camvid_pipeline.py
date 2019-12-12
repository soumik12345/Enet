import numpy as np
import torch, cv2, os
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


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


def decode_segmap(image):
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Pole = [192, 192, 128]
    Road_marking = [255, 69, 0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrian = [64, 64, 0]
    Bicyclist = [0, 128, 192]

    label_colours = np.array([Sky, Building, Pole, Road_marking, Road, 
                              Pavement, Tree, SignSymbol, Fence, Car, 
                              Pedestrian, Bicyclist]).astype(np.uint8)
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


def predict_rgb(model, tensor):
    with torch.no_grad():
        out = model(tensor.float()).squeeze(0)
    out = out.data.max(0)[1].cpu().numpy()
    return decode_segmap(out)


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
	save_every, save_location, save_prefix, epochs):
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
	for epoch in range(1, epochs + 1):
		print('Epoch {}\n'.format(epoch))
		# Training
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
	print(
        '\nTraining Done.\nTraining Mean Loss: {:6f}\nValidation Mean Loss: {:6f}'.format(
            sum(train_loss_history) / epochs,
            sum(val_loss_history) / epochs
        )
    )
	return train_loss_history, val_loss_history