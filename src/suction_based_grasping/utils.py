import numpy as np
import matplotlib.pylab as plt

def read_split_file(split_filename):
	"""
	Loads image names based on a split file
 	Inputs:
		split_filename = filename of split text file (e.g. "train-split.txt")
	Output:
		returns image names (stripped lines from split text file)
	"""
	data_path = '/home/shared/project/data'
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
	data_path = '/home/shared/project/data'
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

