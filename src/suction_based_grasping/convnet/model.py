import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model # not using Sequential currently
from keras.layers import *
import keras.backend as K # for custom loss function

# NOTE: assumes notebook is run from project directory
from src.suction_based_grasping.utils import *

K.set_image_data_format('channels_last') # just to make sure

def heatmap_loss(y_true, y_pred):
	"""
	Calculates the loss of a 3-channel tensor relative to a 1-channel heatmap
	Inputs:
		y_true = (H, W, 1)
		y_pred = (H, W, 3)
	Outptuts:
		loss = scalar loss
	"""
	loss = 0

	# restructure this to use cond() statements or somehow vectorize in different way
	if K.any(y_true==0):
		class_a_scores = K.tf.gather_nd(y_pred, K.tf.where(y_true==0))
		class_a_loss = -K.log(K.exp(class_a_scores[:, 0]) / K.sum(K.exp(class_a_scores), axis=1)) # class a loss
		loss += K.sum(class_a_loss)

	if K.any(y_true==1):
		class_b_scores = K.tf.gather_nd(y_pred, K.tf.where(y_true==1))
		class_b_loss = -K.log(K.exp(class_b_scores[:, 1]) / K.sum(K.exp(class_b_scores), axis=1)) # class b loss
		loss += K.sum(class_b_loss)

	if K.any(y_true==2):
		class_c_scores = K.tf.gather_nd(y_pred, K.tf.where(y_true==2))
		class_c_loss = -K.log(K.exp(class_c_scores[:, 2]) / K.sum(K.exp(class_c_scores), axis=1)) # class c loss
		loss += K.sum(class_c_loss)

	return loss

def init_model(image_shape):
	"""
	Initializes the combined model from two ResNet50 subnets pretrained on imagenet
	"""
	H, W = image_shape
	# ResNet50 rgb model
	resnet_color = ResNet50(include_top=False, pooling=None, input_shape=(H, W, 3), weights='imagenet')
	for layer in resnet_color.layers:
		layer.name = layer.name + '_color'
		layer.trainable = False
	
	# ResNet50 ddd model
	resnet_depth = ResNet50(include_top=False, pooling=None, input_shape=(H, W, 3), weights='imagenet')
	for layer in resnet_depth.layers:
		layer.name = layer.name + '_depth'
		layer.trainable = False
	
	# model with parallel ResNet models and merged output
	merged_out = Concatenate(name='Addydoo')([resnet_color.output, resnet_depth.output]) # merged output from rgb and ddd resnets
	x = Conv2D(512, 1, strides=1, padding='same', activation='relu', use_bias=True, name='Frank')(merged_out)
	x = Conv2D(128, 1, strides=1, padding='same', activation='relu', use_bias=True, name='Bob')(x)
	x = Conv2D(3, 1, strides=1, padding='same', activation='softmax', use_bias=True, name='Joe')(x)
	
	combined_model = Model([resnet_color.input, resnet_depth.input], x)
	combined_model.compile('adam', loss=heatmap_loss, metrics=['accuracy'])

	return combined_model