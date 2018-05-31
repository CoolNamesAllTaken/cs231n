import numpy as np
import matplotlib.pylab as plt

#NOTE: assumes notebook is run from project directory
from src.suction_based_grasping.baseline.test import *

def evaluate():
	data_path = 'home/shared/project/data'
	filename = "{}/{}".format(data_path, "test-split.txt")
	with open(filename, 'r') as file:
		lines = file.readlines()
	test_split = [line.strip() for line in lines]

	results = test()
	for img_name in test_split:
		print("Evaluating image {}".format(img_name))
		sample_result = 
