import numpy as numpy
import matplotlib.pylab as plt

def test():
	data_path = '/home/jkailimcnelly/project/data'
	filename = "{}/{}".format(data_path, "test-split.txt")
	with open(filename, 'r') as file:
		lines = file.readlines()
	test_split = [line.strip() for line in lines]
	
	for img_name in test_split:
		input_color = plt.imread("{}/color-input/{}.png".format(data_path, img_name))
		input_depth = plt.imread("{}/depth-input/{}.png".format(data_path, img_name)) / 10000
		background_color = plt.imread("{}/color-background/{}.png".format(data_path, img_name))
		background_depth = plt.imread("{}/depth-background/{}.png".format(data_path, img_name)) / 10000
		return input_color, input_depth, background_color, background_depth


