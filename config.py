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


CAMVID_CONFIGS = {
	'class_colors' : {
		'obj0' : [255, 0, 0], # Sky
		'obj1' : [0, 51, 204], # Building
		'obj2' : [0, 255, 255], # Posts
		'obj3' : [153, 102, 102], # Road
		'obj4' : [51, 0, 102], # Pavement
		'obj5' : [0, 255, 0], # Trees
		'obj6' : [102, 153, 153], # Signs
		'obj7' : [204, 0, 102], # Fence
		'obj8' : [102, 0, 0], # Car
		'obj9' : [0, 153, 102], # Pedestrian
		'obj10' : [255, 255, 255], # Cyclist
		'obj11' : [0, 0, 0] # bicycles
	}
}