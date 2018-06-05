import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import *

# NOTE: assumes notebook is run from project directory
from src.suction_based_grasping.utils import *

def init_model(image_shape):
	"""
	Initializes the combined model from two ResNet50 subnets pretrained on imagenet
	"""
	H, W = image_shape
	# ResNet50 rgb model
	resnet_color = ResNet50(include_top=False, pooling=None, input_shape=(H, W, 3), weights='imagenet')
	
	# ResNet50 ddd model
	resnet_depth = ResNet50(include_top=False, pooling=None, input_shape=(H, W, 3), weights='imagenet')
	for layer in resnet_depth.layers:
		layer.name = layer.name + '_depth'
	
	# model with parallel ResNet models and merged output
	merged_out = Add()([resnet_color.output, resnet_depth.output]) # merged output from rgb and ddd resnets
	resnet_rgbddd = Model([resnet_color.input, resnet_depth.input], merged_out)
	x = Conv2D(12, 4, strides=1, padding='same', activation='relu', use_bias=True)(resnet_rgbddd.output)
	x = Conv2D(12, 4, strides=1, padding='same', activation='relu', use_bias=True)(x)
	merged_out = Conv2D(12, 4, strides=1, padding='same', activation='softmax', use_bias=True)(x)
	
	combined_model = Model([resnet_color.input, resnet_depth.input], merged_out)

	return combined_model

def train_model(model, X_train, y_train, epochs=20, batch_size=128):
	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size) # TODO: move this to jupyter
	# TODO: use validation part of fit, throw away old stuff