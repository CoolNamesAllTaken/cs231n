import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model # not using Sequential currently
from keras.layers import *
import keras.backend as kb # for custom loss function

# NOTE: assumes notebook is run from project directory
from src.suction_based_grasping.utils import *

def heatmap_loss(y_true, y_pred):
	"""
	Calculates the loss of a 3-channel tensor relative to a 1-channel heatmap
	Inputs:
		y_true = (H, W)
		y_pred = (H, W, 3)
	Outptuts:
		loss = scalar loss
	"""
	loss = 0

	class_a_scores = kb.gather(y_pred, kb.tf.where(y_true==0))
	class_a_loss = -kb.log(kb.exp(class_a_scores[:, 0]) / kb.sum(kb.exp(class_a_scores), axis=1)) # class a loss
	loss += kb.sum(class_a_loss)

	class_b_scores = kb.gather(y_pred, kb.tf.where(y_true==1))
	class_b_loss = -kb.log(kb.exp(class_b_scores[:, 1]) / kb.sum(kb.exp(class_b_scores), axis=1)) # class b loss
	loss += kb.sum(class_b_loss)

	class_c_scores = kb.gather(y_pred, kb.tf.where(y_true==2))
	class_c_loss = -kb.log(kb.exp(class_c_scores[:, 2]) / kb.sum(kb.exp(class_c_scores), axis=1)) # class c loss
	loss += kb.sum(class_c_loss)

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
	
	# ResNet50 ddd model
	resnet_depth = ResNet50(include_top=False, pooling=None, input_shape=(H, W, 3), weights='imagenet')
	for layer in resnet_depth.layers:
		layer.name = layer.name + '_depth'
	
	# model with parallel ResNet models and merged output
	merged_out = Add(name='Addydoo')([resnet_color.output, resnet_depth.output]) # merged output from rgb and ddd resnets
	x = Conv2D(512, 1, strides=1, padding='same', activation='relu', use_bias=True, name='Frank')(merged_out)
	x = Conv2D(128, 1, strides=1, padding='same', activation='relu', use_bias=True, name='Bob')(x)
	x = Conv2D(3, 1, strides=1, padding='same', activation='softmax', use_bias=True, name='Joe')(x)
	
	combined_model = Model([resnet_color.input, resnet_depth.input], x)
	combined_model.compile('adam', loss=heatmap_loss, metrics=['accuracy'])

	return combined_model

def train_model(model, X_train, y_train, epochs=20, batch_size=128):
	model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size) # TODO: move this to jupyter
	# TODO: use validation part of fit, throw away old stuff