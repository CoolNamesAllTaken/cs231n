import numpy as np
import matplotlib.pylab as plt

#NOTE: assumes notebook is run from project directory
from src.suction_based_grasping.baseline.test import *
from src.suction_based_grasping.utils import *

def evaluate(num_imgs=-1):
	# parse test split from dataset, load baseline suction prediction results
	test_split = read_split_file("test-split.txt")
	if num_imgs > 0:
		test_split = test_split[0:num_imgs]
	
	results = test(num_imgs=num_imgs, display_labels=True)
	
	#loop through all test samples and evaluate baseline suction prediction
	#results against ground truth manual annotations
	sum_tp, sum_fp, sum_tn, sum_fn = 0, 0, 0, 0	
	for img_name in test_split:
		print("Evaluating image {}".format(img_name))
		sample_result = results[img_name]
		
		#load ground thruth manual annotations for suction affordances
		#0 - negative, 128 - positive, 255 - neutral (no loss)
		sample_label = load_label(img_name)
		
		#suction affordance threshold
		threshold = np.max(sample_result) - 0.2 # top ?? prediction

		#compute errors
		print(np.unique(sample_label))
		print(np.any(sample_label == 128))
		sample_tp = np.logical_and(sample_result > threshold, sample_label == 0)
		sample_fp = np.logical_and(sample_result > threshold, sample_label == 1)
		sample_tn = np.logical_and(sample_result <= threshold, sample_label == 1)
		sample_fn = np.logical_and(sample_result <= threshold, sample_label == 0)
		sum_tp += np.sum(sample_tp)
		sum_fp += np.sum(sample_fp)
		sum_tn += np.sum(sample_tn)
		sum_fn += np.sum(sample_fn)

	print("sum_tp={} sum_fp={} sum_tn={} sum_fn={}".format(sum_tp, sum_fn, sum_tn, sum_fn))
	precision = float(sum_tp) / (sum_tp + sum_fp)
	recall = float(sum_tp) / (sum_tp + sum_fn)
	print('Precision: {}'.format(precision))
	print('Recall: {}'.format(recall))
	
