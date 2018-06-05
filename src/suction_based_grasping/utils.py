import numpy as np
import matplotlib.pylab as plt
import skimage
from skimage import transform # for image resizing

data_path = '/home/shared/project/data'

def read_split_file(split_filename):
	"""
	Loads image names based on a split file
 	Inputs:
		split_filename = filename of split text file (e.g. "train-split.txt")
	Output:
		returns image names (stripped lines from split text file)
	"""
	filename = "{}/{}".format(data_path, split_filename)
	with open(filename, 'r') as file:
		lines = file.readlines()
	img_names = [line.strip() for line in lines]
	return img_names

def load_image(img_name, display=False):
	"""
	Loads a set of images corresponding to an image name
	Inputs:
		img_name = name of image to read
		display = True if images and camera intrinsics should be displayed, False otherwise
	Outputs:
		returns a tuple of input_color, input_depth, background_color, background_depth, camera_intrinsics
	"""
	input_color = plt.imread("{}/color-input/{}.png".format(data_path, img_name))
	input_depth = plt.imread("{}/depth-input/{}.png".format(data_path, img_name)) / 10000
	background_color = plt.imread("{}/color-background/{}.png".format(data_path, img_name))
	background_depth = plt.imread("{}/depth-background/{}.png".format(data_path, img_name)) / 10000
	camera_intrinsics = np.loadtxt("{}/camera-intrinsics/{}.txt".format(data_path, img_name))

	if display:
		plt.figure()
		plt.imshow(input_color)
		plt.axis('off')

		plt.figure()
		plt.imshow(input_depth)
		plt.axis('off')

		plt.figure()
		plt.imshow(background_color)
		plt.axis('off')

		plt.figure()
		plt.imshow(background_depth)
		plt.axis('off')

		print(camera_intrinsics)

	return input_color, input_depth, background_color, background_depth, camera_intrinsics

def load_label(img_name, display=False):
	"""
	Loads a label image for a given image name
	Inputs:
		img_name = name of image label to load
		display = whether to display the image label
	Outputs:
		returns image label
	"""
	label = plt.imread("{}/label/{}.png".format(data_path, img_name))

	if display:
		plt.figure()
		plt.imshow(label)
		plt.axis('off')

	return label

def load_images_from_list(img_names, target_shape, verbose):
	"""
	Helper function for train_test_split
	"""
	N = len(img_names)
	H, W = target_shape
	C = 6 # RGBDDD
	X = np.empty((N, H, W, C))
	y = np.empty((N, H, W))
	for i in range(N):
		if verbose: print("{}/{}\t{:0.2f}%".format(i+1, N, (i+1)/N*100), end='\r')
		# load image (RGBD), ignore background RGBD and camera instrinsics
		input_color, input_depth, _, _, _ = load_image(img_names[i])
		input_color = skimage.transform.resize(input_color, (H, W, 3))
		input_depth = skimage.transform.resize(input_depth, (H, W, 1))
		X[i, :, :, 0:3] = input_color
		X[i, :, :, 3:6] = np.repeat(input_depth, 3, axis=2)
		label = load_label(img_names[i])
		label = skimage.transform.resize(label, (H, W))
		y[i, :, :] = label
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

def train_val_split(X_train_all, y_train_all, train_frac):
	"""
	Splits a training data set into training and validation sets based on a train-val split equal to train_frac
	Inputs:
		x_train_all = full training set inputs
		y_train_all = full training set labels
	Outputs:
		X_train = training set inputs
		X_val = validation set inputs
		y_train = training set labels
		y_val = validation set labels
	"""
	N = X_train_all.shape[0]
	num_train = int(train_frac * N)
	num_val = N - num_train
	train_inds = range(num_train)
	val_inds = range(num_train, num_train+num_val)
	X_train = X_train_all[train_inds]
	y_train = y_train_all[train_inds]
	X_val = X_train_all[val_inds]
	y_val = y_train_all[val_inds]
	return X_train, X_val, y_train, y_val