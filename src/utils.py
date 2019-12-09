import numpy as np


def intersection(prediction, ground_truth, n_classes):
	'''Find Intersection Area
	Params:
		prediction	 -> Prediction
		ground_truth -> Ground Truth
		n_classes	 -> Number of classes
	'''
	prediction = np.asarray(
		prediction,
		dtype=np.uint8
	).copy()
	ground_truth = np.asarray(
		ground_truth,
		dtype=np.uint8
	).copy()
	prediction += 1
	ground_truth += 1
	prediction = prediction * (ground_truth > 0)
	intersection = prediction * (prediction == ground_truth)
	(area_intersection, _) = np.histogram(
		intersection,
		bins=n_classes,
		range=(1, n_classes)
	)
	return area_intersection


def union(prediction, ground_truth, n_classes):
	'''Find Union Area
	Params:
		prediction	 -> Prediction
		ground_truth -> Ground Truth
		n_classes	 -> Number of classes
	'''
	prediction = np.asarray(
		prediction,
		dtype=np.uint8
	).copy()
	ground_truth = np.asarray(
		ground_truth,
		dtype=np.uint8
	).copy()
	prediction += 1
	ground_truth += 1
	prediction = prediction * (ground_truth > 0)
	intersection = prediction * (prediction == ground_truth)
	(area_intersection, _) = np.histogram(
		intersection,
		bins=n_classes,
		range=(1, n_classes)
	)
	(area_prediction, _) = np.histogram(
		prediction,
		bins=n_classes,
		range=(1, n_classes)
	)
	(area_gt, _) = np.histogram(
		ground_truth,
		bins=n_classes,
		range=(1, n_classes)
	)
	return area_prediction + area_gt - area_intersection