import numpy as np
import matplotlib.pylab as plt

# NOTE: assumes notebook is run from project directory
from src.suction_based_grasping.baseline.predict import *
from src.suction_based_grasping.utils import *

def test(num_imgs=-1, display_labels=False):
	test_split = read_split_file("test-split.txt")
	if num_imgs > 0:
		test_split = test_split[0:num_imgs]
	results = {}
	for img_name in test_split:
		print("Working on image {}".format(img_name))

		input_color, input_depth, background_color, background_depth, camera_intrinsics = load_image(img_name)

		plt.figure(figsize=(20, 50))
		plt.subplot(1, 4, 1)
		plt.imshow(input_color)
		plt.axis('off')

		plt.subplot(1, 4, 2)
		plt.imshow(input_depth)
		plt.axis('off')

		affordance_map, surface_normal_map = predict(input_color, input_depth, background_color, background_depth, camera_intrinsics)

		plt.subplot(1, 4, 3)
		plt.imshow(affordance_map)
		plt.axis('off')

		if display_labels:
			label = load_label(img_name)
			plt.subplot(1, 4, 4)
			plt.imshow(label)
			plt.axis('off')

		# get plots to show live
		plt.pause(0.05)
		plt.show()

		results[img_name] = affordance_map

	return results



