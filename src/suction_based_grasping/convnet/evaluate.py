import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from src.suction_based_grasping.utils import *
from src.suction_based_grasping.convnet.data_utils import *

def evaluate(model, X_target_shape=(480,640), y_target_shape=(480,640), num_imgs=-1):
	# parse test split from dataset, load convnet suction prediction results
	img_names = read_split_file("test-split.txt")
	if num_imgs > 0:
		img_names= img_names[0:num_imgs]
	
	X_test, y_test = load_images_from_list(img_names, X_target_shape, y_target_shape, verbose=True)
	heatmaps = model.predict(X_test) # (N, H, W, 3)

	heatmap_affordances = np.argmax(heatmaps, axis=3)
	y_test_affordances = np.argmax(y_test, axis=3)
	
	sum_tp = np.sum(np.logical_and(heatmap_affordances>0, y_test_affordances>0)) # true positives
	sum_fp = np.sum(np.logical_and(heatmap_affordances>0, y_test_affordances==0)) # false positives
	sum_tn = np.sum(np.logical_and(heatmap_affordances==0, y_test_affordances==0)) # true negatives
	sum_fn = np.sum(np.logical_and(heatmap_affordances==0, y_test_affordances>0)) # false negatives

	print("sum_tp={} sum_fp={} sum_tn={} sum_fn={}".format(sum_tp, sum_fn, sum_tn, sum_fn))
	precision = float(sum_tp) / (sum_tp + sum_fp)
	recall = float(sum_tp) / (sum_tp + sum_fn)
	print('Precision: {}'.format(precision))
	print('Recall: {}'.format(recall))
