# Based on andyzeng/arc-robot-vision/suction-based-grasping/baseline/predict.m
# https://github.com/andyzeng/arc-robot-vision/blob/master/suction-based-grasping/baseline/predict.m

import numpy as np
import pcl.PointCloud() as pc

# A baseline algorithm for predicting affordances for suction-based
# grasping: (1) compute 3D surface normals of the point cloud projected
# from the RGB-D image (2) measure the variance of the surface normals
# where higher variance = lower affordance.
#
# function [affordanceMap,surfaceNormalsMap] = predict(inputColor,inputDepth,backgroundColor,backgroundDepth,cameraIntrinsics)
# Input:
#   inputColor         - 480x640x3 float array of RGB color values scaled to range [0,1]
#   inputDepth         - 480x640 float array of depth values in meters
#   backgroundColor    - 480x640x3 float array of RGB color values scaled to range [0,1]
#   backgroundDepth    - 480x640 float array of depth values in meters
#   cameraIntrinsics   - 3x3 camera intrinsics matrix
# Output:
#   affordanceMap      - 480x640 float array of affordance values in range [0,1]
#   surfaceNormalsMap  - 480x640x3 float array of surface normals in camera coordinates (meters)

def predict(inputColor, inputDepth, backgroundColor, backgroundDepth, cameraIntrinsics):
	# Scale color images between [0, 1]
	inputColor /= 255.
	backgroundColor /= 255.
	
	# Do background subtraction to get foreground mask
	foregroundMaskColor = !(np.sum(np.abs(inputColor - backgroundColor) < 0.3, axis=3) == 3)
	foregroundMaskDepth = backgroundDepth != 0 and np.abs(inputDepth - backgroundDepth) > 0.02
	foregroundMask = foregroundMaskColor or foregroundMaskDepth

	# Project depth into camera space
	[pixX, pixY] = np.meshgrid(list(range(1, 641)), list(range(1, 481)))
	camX = (pixX - cameraIntrinsics[1, 3]). * inputDepth / cameraIntrinsics[1, 1]
	camY = (pixY - cameraIntrinsics[2, 3]). * inputDepth / cameraIntrinsics[2, 2]
	camZ = inputDepth

	# Only use points with valid depth and within foreground mask
	validDepth = foregroundMask and camZ != 0
	inputPoints = [camX[validDepth], camY[validDepth], camZ[validDepth]]
	
	# Get foreground point cloud normals
	foregroundPointcloud = pc.from_list(inputPoints)
		 
