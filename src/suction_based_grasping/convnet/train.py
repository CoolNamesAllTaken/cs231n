import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# NOTE: assumes notebook is run from project directory
from src.suction_based_grasping.utils import *

def train():
	print("started train")
	X_train, X_test, y_train, y_test = train_test_split("train-split.txt", "test-split.txt")
	print(len(X_train), len(X_test), len(y_train), len(y_test))