import numpy as np
import matplotlib.pylab as plt

#NOTE: assumes notebook is run from project directory
from src.suction_based_grasping.baseline.test import *
from src.suction_based_grasping.utils import *

def evaluate():
	# parse test split from dataset, load baseline suction prediction results
	test_split = read_split_file("test-split.txt")
	results = test()
	
	#loop through all test samples and evaluate baseline suction prediction
	#results against ground truth manual annotations
	sum_tp, sum_fp, sum_tn, sum_fn = 0, 0, 0, 0	
	for img_name in test_split:
		print("Evaluating image {}".format(img_name))
		sample_result = results[img_name]
		
		#load ground thruth manual annotations for suction affordances
		#0 - negative, 128 - positive, 255 - neutral (no loss)
		sample_label = load_label(image_name)
		
		#suction affordance threshold
		threshold = np.max(sample_result) - 0.0001 #top 1 prediction

		#compute errors
		sample_tp = (sample_result > threshold and sample_label == 128)
		sample_fp = (sample_result > threshold and sample_label == 0)
		sample_tn = (sample_result <= threshold and sample_label == 0)
		sample_fn = (sample_result <= threshold and sample_label == 128)		sum_tp += np.sum(sample_tp)
		sum_fp += np.sum(sample_fp)
		sum_tn += np.sum(sample_tn)
		sum_fn += np.sum(sample_fn)

	precision = sum_tp / (sum_tp + sum_fp)
	recall = sum_tp / (sum_tp + sum_fn)
	print('Precision:', precision)
	print('Recall:', recall)
	
