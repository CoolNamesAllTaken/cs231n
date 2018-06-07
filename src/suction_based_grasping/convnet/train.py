import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from src.suction_based_grasping.utils import *
from src.suction_based_grasping.convnet.model import *
from src.suction_based_grasping.convnet.data_utils import *

from keras import models # for load_model

X_target_shape = (960, 1280) # height and width of inputs to ResNet50
y_target_shape = (480, 640)
model_filepath = '/home/shared/project/src/suction_based_grasping/convnet/models'

train_batch_size = 2
val_batch_size = 2
validation_split = 0.2

def train():
	model = init_model(X_target_shape, y_target_shape)

	img_names = read_split_file("train-split.txt")
	N = len(img_names)
	N_train = int((1-validation_split)*N)
	N_val = N-N_train
	train_img_names = img_names[0:N_train]
	val_img_names = img_names[N_train:N]

	model.fit_generator(
	    image_generator(train_img_names, train_batch_size, X_target_shape=X_target_shape, y_target_shape=y_target_shape),
	    validation_data=image_generator(val_img_names, val_batch_size, X_target_shape=X_target_shape, y_target_shape=y_target_shape),
	    validation_steps=1,
	    steps_per_epoch=5, epochs=10, verbose=2)