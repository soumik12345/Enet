from config import *
from tqdm import tqdm
from random import randint
from .model import DeepLabV3
from torch.nn import CrossEntropyLoss
from config import TRAINING_PARAMETERS
from torch.optim import Adam, SGD, lr_scheduler


def train(model, data_loader, device, epochs):
	'''Train DeepLabV3 Model
	Params:
		model		-> Segmentation Model
		data_loader	-> Train Data Loader
		epochs		-> Number of epochs
	'''
	criterion = CrossEntropyLoss()
	adam = Adam(
		model.parameters(),
		TRAINING_PARAMETERS['learning_rate'],
		weight_decay=TRAINING_PARAMETERS['weight_decay']
	)
	learning_rate_sheduler = lr_scheduler.LambdaLR(
		adam,
		lambda epoch: (1 - (epoch / epochs)) ** 0.9
	)
	train_loss_history = []
	for epoch in range(1, epochs + 1):
		print('Epoch {}\n'.format(epoch))
		train_loss = 0
		learning_rate_sheduler.step()
		model.train()
		for step in tqdm(range(2975 // TRAINING_PARAMETERS['batch_size'])):
			x_batch, y_batch = next(iter(data_loader))
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)
			adam.zero_grad()
			output = model(x_batch)
			loss = criterion(output, y_batch)
			loss.backward()
			adam.step()
			train_loss += loss.item()
		train_loss_history.append(train_loss)
		print('Loss: {}\n'.format(train_loss_history[-1]))
	return train_loss_history