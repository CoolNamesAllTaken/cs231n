import numpy as np
import matplotlib.pylab as plt

# NOTE: assumes notebook is run from project directory
from src.suction_based_grasping.baseline.predict import *
from src.suction_based_grasping.utils import *

def test():
	test_split = read_split_file("test-split.txt")
	
	results = {}
	for img_name in test_split:
		print("Working on image {}".format(img_name))

		input_color, input_depth, background_color, background_depth, camera_intrinsics = load_image(img_name)

		plt.figure()
		plt.imshow(input_color)
		plt.axis('off')

		plt.figure()
		plt.imshow(input_depth)
		plt.axis('off')

		affordance_map, surface_normal_map = predict(input_color, input_depth, background_color, background_depth, camera_intrinsics)

		plt.figure()
		plt.imshow(affordance_map)
		plt.axis('off')

		# get plots to show live
		plt.pause(0.05)
		plt.show()

		results[img_name] = affordance_map

	return results



