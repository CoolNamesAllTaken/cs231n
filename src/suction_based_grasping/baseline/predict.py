# Based on andyzeng/arc-robot-vision/suction-based-grasping/baseline/predict.m
# https://github.com/andyzeng/arc-robot-vision/blob/master/suction-based-grasping/baseline/predict.m

import numpy as np
import pcl

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
	[pixX, pixY] = np.meshgrid(list(range(640)), list(range(480)))
	camX = (pixX - cameraIntrinsics[0, 2]). * inputDepth / cameraIntrinsics[0, 0]
	camY = (pixY - cameraIntrinsics[1, 2]). * inputDepth / cameraIntrinsics[1, 1]
	camZ = inputDepth

	# Only use points with valid depth and within foreground mask
	validDepth = foregroundMask and camZ != 0
	inputPoints = [camX[validDepth], camY[validDepth], camZ[validDepth]]
	
	# Get foreground point cloud normals
	foregroundPointcloud = pcl.PointCloud().from_list(inputPoints)
	foregroundNormals = foregroundPointcloud.calc_normals(50)

	# Flip normals to point toward sensor
	sensorCenter = [0, 0, 0]
	inputPoints = inputPoints.T
	for k in range(len(inputPoints[2])):
		p1 = sensorCenter - [inputPoints[0, k], inputPoints[1, k], inputPoints[2, k]]
		p2 = [foregroundNormals[k, 0], foregroundNormals[k, 1], foregroundNormals[k, 2]]
		angle = np.arctan2(p1.dot(p2.T), np.linalg.norm(np.cross(p1, p2)))
		if angle <= np.pi / 2 and angle >= -np.pi / 2:
			foregroundNormals[k] *= -1

	# Project normals back to image plane
	pixX = math.round(inputPoints[0] * cameraIntrinsics[0, 0]. / inputPoints[2] + cameraIntrinsics[0, 2])
	pixY = math.round(inputPoints[1] * cameraIntrinsics[1, 1]. / inputPoints[2] + cameraIntrinsics[1, 2])
	surfaceNormalsMap = np.zeros_like(inputColor)
	surfaceNormalsMap[np.ravel_multi_index(surfaceNormalsMap.size, (pixY, pixX, np.zeros_like(pixY))] = foregroundNormals[:, 0]
	surfaceNormalsMap[np.ravel_multi_index(surfaceNormalsMap.size, (pixY, pixX, np.ones_like(pixY))] = foregroundNormals[:, 1]	
	surfaceNormalsMap[np.ravel_multi_index(surfaceNormalsMap.size, (pixY, pixX, 2 * np.ones_like(pixY))] = foregroundNormals[:, 2]
