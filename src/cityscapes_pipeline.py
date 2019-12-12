import torch
import numpy as np
from glob import glob
from PIL import Image
from torch import LongTensor
from random import random, randint
from config import CITYSCAPES_CONFIG
from torch.utils.data import Dataset
from torch.nn import ConstantPad2d, ZeroPad2d
from torchvision.transforms import Compose, ToTensor, Normalize


def preprocess(image, mask):
    # Apply Flip
    if random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    # Image Transforms
    _transforms = Compose([
        ToTensor(),
        Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    image = _transforms(image)
    # Transform mask to Tensor
    mask = LongTensor(np.array(mask).astype(np.int64))
    # Apply Crop
    crop = CITYSCAPES_CONFIG['crop']
    if crop:
        h, w = image.shape[1], image.shape[2]
        pad_vertical = max(0, crop[0] - h)
        pad_horizontal = max(0, crop[1] - w)
        image = ZeroPad2d((
            0, pad_horizontal,
            0, pad_vertical
        ))(image)
        mask = ConstantPad2d((
                0, pad_horizontal,
                0, pad_vertical
            ),
        255)(mask)
        h, w = image.shape[1], image.shape[2]
        i = randint(0, h - crop[0])
        j = randint(0, w - crop[1])
        image = image[:, i : i + crop[0], j : j + crop[1]]
        mask = mask[i : i + crop[0], j : j + crop[1]]
    return image, mask



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