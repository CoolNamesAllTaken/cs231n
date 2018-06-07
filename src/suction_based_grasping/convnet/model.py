import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model # not using Sequential currently
from keras.layers import *
from keras.constraints import max_norm
from keras import losses # for custom loss function
from keras import optimizers # for model compilation
import keras.backend as K # for custom loss function

# NOTE: assumes notebook is run from project directory
from src.suction_based_grasping.utils import *

K.set_image_data_format('channels_last') # just to make sure

def heatmap_loss(y_true, y_pred):
	"""
	Calculates the loss of a 3-channel tensor relative to a 1-channel heatmap
	Inputs:
		y_true = (N, H, W, 3)
		y_pred = (N, H, W, 3)
	Outptuts:
		loss = scalar loss
	"""
	loss = 0
	N = K.shape(y_pred)[0]
	H = K.shape(y_pred)[1]
	W = K.shape(y_pred)[2]
	y_true_flat = K.reshape(y_true, [N, -1, 3]) # semi-flatten ground truth
	y_pred_flat  = K.reshape(y_pred, [N, -1, 3]) # semi-flatten network output
	loss = losses.categorical_crossentropy(y_true_flat, y_pred_flat)

	return loss

def init_model(X_target_shape, y_target_shape):
	"""
	Initializes the combined model from two ResNet50 subnets pretrained on imagenet
	"""
	H_X, W_X = X_target_shape
	H_y, W_y = y_target_shape # not used

	# ResNet50 rgb model
	resnet_color = ResNet50(include_top=False, pooling=None, input_shape=(H_X, W_X, 3), weights='imagenet')
	for layer in resnet_color.layers:
		layer.name = layer.name + '_color'
		layer.trainable = False
	resnet_color = Model(resnet_color.inputs, resnet_color.layers[-2].output) # remove last layer
	
	# ResNet50 ddd model
	resnet_depth = ResNet50(include_top=False, pooling=None, input_shape=(H_X, W_X, 3), weights='imagenet')
	for layer in resnet_depth.layers:
		layer.name = layer.name + '_depth'
		layer.trainable = False
	resnet_depth = Model(resnet_depth.inputs, resnet_depth.layers[-2].output) # remove last layer

	# model with parallel ResNet models and merged output
	x = Concatenate(name='Addydoo')([resnet_color.output, resnet_depth.output]) # merged output from rgb and ddd resnets (N, 15, 20, 3)
	x = UpSampling2D(size=(4, 4))(x) # upsample to (N, 60, 80, 3)
	x = Conv2D(512, 1, strides=1, padding='same', activation='relu', use_bias=True, name='Frank')(x)
	# x = Dropout(0.2)(x)
	x = UpSampling2D(size=(2, 2))(x) # upsample to (N, 120, 160, 3)
	x = Conv2D(128, 1, strides=1, padding='same', activation='relu', use_bias=True, name='Bob')(x)
	# x = Dropout(0.2)(x)
	x = UpSampling2D(size=(2, 2))(x) # upsample to (N, 240, 320, 3)
	x = Conv2D(3, 1, strides=1, padding='same', activation='softmax', use_bias=True, name='Joe')(x)
	x = UpSampling2D(size=(2, 2))(x) # upsample to (N, 480, 640, 3)
	
	combined_model = Model([resnet_color.input, resnet_depth.input], x)
	optimizer = optimizers.SGD(lr=0.001, momentum=0.99, decay=0.0)
	combined_model.compile(optimizer, loss=heatmap_loss, metrics=['accuracy'])

	return combined_model
