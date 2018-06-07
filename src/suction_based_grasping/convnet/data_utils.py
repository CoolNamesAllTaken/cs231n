import numpy as np
import matplotlib.pylab as plt
import skimage
from skimage import transform # for image resizing

from keras.utils.np_utils import to_categorical # for transforming y data

from src.suction_based_grasping.utils import * # assumes it's being run from project directory

def preprocess_color_img(color_img, X_target_shape):
	H, W = X_target_shape
	color_img = skimage.transform.resize(color_img, (H, W, 3))
	color_img -= np.mean(color_img, axis=(0, 1))
	color_img /= (np.std(color_img, axis=(0, 1)) + 1e-8)
	return color_img

def preprocess_depth_img(depth_img, X_target_shape):
	H, W = X_target_shape
	depth_img = skimage.transform.resize(depth_img, (H, W, 1))
	depth_img *= 65536./10000 # why is this a thing
	depth_img = np.clip(depth_img, 0.0, 1.2) # depth range of Intel RealSense SR300
	depth_img -= np.mean(depth_img)
	depth_img /= (np.std(depth_img) + 1e-8)
	depth_img = np.repeat(depth_img, 3, axis=2)
	return depth_img

def preprocess_label(label, y_target_shape):
	H, W = y_target_shape
	label = skimage.transform.resize(label, (H, W))
	label *= 2 # entries = [0, ??, 2]
	label = np.around(label) # entries = [0, 1, 2]
	label = np.reshape(label, (-1,)) # flatten label before to_categorical
	label = to_categorical(label, num_classes=3)
	label = np.reshape(label, (H, W, 3)) # 2D matrix of 1-hot probability distributions
	return label

def load_images_from_list(img_names, X_target_shape, y_target_shape, verbose=False):
	"""
	Helper function for train_test_split
	"""
	N = len(img_names)
	H_X, W_X = X_target_shape
	H_y, W_y = y_target_shape
	X_color = np.empty((N, H_X, W_X, 3))
	X_depth = np.empty((N, H_X, W_X, 3))
	y = np.empty((N, H_y, W_y, 3))
	for i in range(N):
		if verbose: print("{}/{}\t{:0.2f}%".format(i+1, N, (i+1)/N*100), end='\r')
		# load image (RGBD), ignore background RGBD and camera instrinsics
		input_color, input_depth, background_color, background_depth, _ = load_image(img_names[i])
		X_color[i, :, :, :] = preprocess_color_img(input_color-background_color, X_target_shape)
		X_depth[i, :, :, :] = preprocess_depth_img(input_depth-background_depth, X_target_shape)
		label = load_label(img_names[i])
		# y[i, :, :, :] = np.expand_dims(preprocess_label(label, X_target_shape), axis=3)
		y[i, :, :, :] = preprocess_label(label, y_target_shape)
		X = [X_color, X_depth]
	return X, y

def train_test_split(train_split_filename, test_split_filename, num_train=-1, num_test=-1, X_target_shape=(480, 640), y_target_shape=(480,640), verbose=True):
	"""
	Returns a training split and test split based on given text files.  Resizes images to be HxW (X_target_shape)
	NOTE: X_target_shape initialized to default input size of ResNet50 (224, 224)
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
	if num_train > 0: train_img_names = train_img_names[0:num_train]
	X_train, y_train = load_images_from_list(train_img_names, X_target_shape, y_target_shape, verbose=verbose)
	if verbose: print("Done!")

	if verbose: print("Loading Test Data from {}".format(test_split_filename))
	test_img_names = read_split_file(test_split_filename)
	if num_test > 0: test_img_names = test_img_names[0:num_test]
	X_test, y_test = load_images_from_list(test_img_names, X_target_shape, y_target_shape, verbose)
	if verbose: print("Done!")

	return X_train, X_test, y_train, y_test

def image_generator(img_names, batch_size, X_target_shape=(960, 1280), y_target_shape=(480,640)):
	"""
	Image loader for Keras.  Try to make batch_size even divisor of total input images.
	"""
	L = len(img_names)

	while True: # make generator infinite for Keras
		batch_start = 0
		batch_end = batch_size
		while batch_start < L:
			limit = min(batch_end, L)
			X, y = load_images_from_list(img_names[batch_start:limit], X_target_shape, y_target_shape, verbose=True)
			yield(X, y) # send over tuple of two numpy arrays with batch_size (or fewer) samples
			batch_start += batch_size
			batch_end += batch_size