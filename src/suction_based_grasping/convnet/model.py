import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

def model_init_fn():
	input_shape = (480, 640, 4) # RGB-D image
	layers = [
	tf.layers.Conv2D
	]