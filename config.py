CITYSCAPES_CONFIG = {
	'folders_map' : {
		'image' : 'leftImg8bit',
		'label' : 'gtFine'
	},
	'data_format' : {
		'image' : 'png',
		'label' : 'png'
	},
	'postfix' : {
		'image': '_leftImg8bit',
		'label': '_gtFine_labelTrainIds'
	},
	'classes' : 35,
	'crop' : (512, 512)
}

PRETRAINED_MODELS = {
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

TRAINING_PARAMETERS = {
	'learning_rate' : 3e-4,
	'weight_decay' : 0.0001,
	'batch_size' : 16,
	'epochs' : 100
}