import numpy as np
import matplotlib.pylab as plt
import skimage
from skimage import transform # for image resizing

from keras.utils.np_utils import to_categorical # for transforming y data

from src.suction_based_grasping.utils import * # assumes it's being run from project directory

def preprocess_color_img(color_img, target_shape):
	H, W = target_shape
	color_img = skimage.transform.resize(color_img, (H, W, 3))
	color_img -= np.mean(color_img, axis=(0, 1))
	color_img /= (np.std(color_img, axis=(0, 1)) + 1e-8)
	return color_img

def preprocess_depth_img(depth_img, target_shape):
	H, W = target_shape
	depth_img = skimage.transform.resize(depth_img, (H, W, 1))
	depth_img *= 65536./10000 # why is this a thing
	depth_img = np.clip(depth_img, 0.0, 1.2) # depth range of Intel RealSense SR300
	depth_img -= np.mean(depth_img)
	depth_img /= (np.std(depth_img) + 1e-8)
	depth_img = np.repeat(depth_img, 3, axis=2)
	return depth_img

def preprocess_label(label, target_shape):
	H, W = target_shape
	label = skimage.transform.resize(label, (H, W))
	label *= 2 # entries = [0, ??, 2]
	label = np.around(label) # entries = [0, 1, 2]
	# label = to_categorical(label, num_classes=3)
	return label

def load_images_from_list(img_names, target_shape, verbose):
	"""
	Helper function for train_test_split
	"""
	N = len(img_names)
	H, W = target_shape
	X_color = np.empty((N, H, W, 3))
	X_depth = np.empty((N, H, W, 3))
	y = np.empty((N, H, W))
	for i in range(N):
		if verbose: print("{}/{}\t{:0.2f}%".format(i+1, N, (i+1)/N*100), end='\r')
		# load image (RGBD), ignore background RGBD and camera instrinsics
		input_color, input_depth, _, _, _ = load_image(img_names[i])
		X_color[i, :, :, :] = preprocess_color_img(input_color, target_shape)
		X_depth[i, :, :, :] = preprocess_depth_img(input_depth, target_shape)
		label = load_label(img_names[i])
		y[i, :, :] = preprocess_label(label, target_shape)
		X = [X_color, X_depth]
	return X, y

def train_test_split(train_split_filename, test_split_filename, target_shape=(224, 224), verbose=True):
	"""
	Returns a training split and test split based on given text files.  Resizes images to be HxW (target_shape)
	NOTE: target_shape initialized to default input size of ResNet50 (224, 224)
	Inputs:
		train_split_filename = name of file with training split filenames
		test_split_filename = name of file with test split filenames
	Outputs:
		X_train = list of training images
		X_test = list of test images
		y_train = list of training image labels (heatmaps)
		y_test = list of test image labels (heatmaps)
	"""
	if verbose: print("Loading Training Data from {}".format(train_split_filename))
	train_img_names = read_split_file(train_split_filename)
	X_train, y_train = load_images_from_list(train_img_names, target_shape, verbose)
	if verbose: print("Done!")

	if verbose: print("Loading Test Data from {}".format(test_split_filename))
	test_img_names = read_split_file(test_split_filename)
	X_test, y_test = load_images_from_list(test_img_names, target_shape, verbose)
	if verbose: print("Done!")

	return X_train, X_test, y_train, y_test