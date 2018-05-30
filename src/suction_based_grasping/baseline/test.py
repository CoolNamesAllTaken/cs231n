import numpy as np
import matplotlib.pylab as plt

# NOTE: assumes notebook is run from project directory
from src.suction_based_grasping.baseline.predict import *

def test():
	data_path = '/home/jkailimcnelly/project/data'
	filename = "{}/{}".format(data_path, "test-split.txt")
	with open(filename, 'r') as file:
		lines = file.readlines()
	test_split = [line.strip() for line in lines]
	
	results = {}
	for img_name in test_split:
		print("Working on image {}".format(img_name))

		input_color = plt.imread("{}/color-input/{}.png".format(data_path, img_name))
		input_depth = plt.imread("{}/depth-input/{}.png".format(data_path, img_name)) / 10000
		background_color = plt.imread("{}/color-background/{}.png".format(data_path, img_name))
		background_depth = plt.imread("{}/depth-background/{}.png".format(data_path, img_name)) / 10000
		camera_intrinsics = np.loadtxt("{}/camera-intrinsics/{}.txt".format(data_path, img_name))

		plt.figure()
		plt.imshow(input_color)
		plt.axis('off')

		plt.figure()
		plt.imshow(input_depth)
		plt.axis('off')

		# plt.figure()
		# plt.imshow(background_color)
		# plt.axis('off')

		# plt.figure()
		# plt.imshow(background_depth)
		# plt.axis('off')

		# print(camera_intrinsics)

		affordance_map, surface_normal_map = predict(input_color, input_depth, background_color, background_depth, camera_intrinsics)

		plt.figure()
		plt.imshow(affordance_map)
		plt.axis('off')

		# get plots to show live
		plt.pause(0.05)
		plt.show()

		results[img_name] = affordance_map

	return results



