# Based on andyzeng/arc-robot-vision/suction-based-grasping/baseline/predict.m
# https://github.com/andyzeng/arc-robot-vision/blob/master/suction-based-grasping/baseline/predict.m

import numpy as np
import pcl
from scipy.ndimage.filters import uniform_filter
import matplotlib.pylab as plt # FOR TESTING ONLY

# From https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
def window_stdev(X, window_size):
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1)

# A baseline algorithm for predicting affordances for suction-based
# grasping: (1) compute 3D surface normals of the point cloud projected
# from the RGB-D image (2) measure the variance of the surface normals
# where higher variance = lower affordance.
#
# function [affordanceMap,surfaceNormalsMap] = predict(inputColor,inputDepth,backgroundColor,backgroundDepth,cameraIntrinsics)
# Input:
#   inputColor         - 480x640x3 float array of RGB color values
#   inputDepth         - 480x640 float array of depth values in meters
#   backgroundColor    - 480x640x3 float array of RGB color values 
#   backgroundDepth    - 480x640 float array of depth values in meters
#   cameraIntrinsics   - 3x3 camera intrinsics matrix
# Output:
#   affordanceMap      - 480x640 float array of affordance values in range [0,1]
#   surfaceNormalsMap  - 480x640x3 float array of surface normals in camera coordinates (meters)

def predict(inputColor, inputDepth, backgroundColor, backgroundDepth, cameraIntrinsics):
	print("Starting predict...")
	# Scale color images between [0, 1]
	# inputColor /= 255.
	# backgroundColor /= 255.

	# plt.figure()
	# plt.imshow(inputColor)
	# plt.axis('off')

	# plt.figure()
	# plt.imshow(inputDepth)
	# plt.axis('off')

	# plt.figure()
	# plt.imshow(backgroundColor)
	# plt.axis('off')

	# plt.figure()
	# plt.imshow(backgroundDepth)
	# plt.axis('off')
	
	# Do background subtraction to get foreground mask
	foregroundMaskColor = (np.sum(np.abs(inputColor - backgroundColor) < 0.3, axis=2) != 3) # mask pixels similar to background
	foregroundMaskDepth = np.logical_and(backgroundDepth != 0, np.abs(inputDepth - backgroundDepth) > 0.02)
	foregroundMask = np.logical_or(foregroundMaskColor, foregroundMaskDepth)

	# show masked image
	# plt.figure()
	# showImg = inputColor
	# showImg[foregroundMask == False] = 0
	# plt.imshow(showImg)
	# plt.axis('off')

	# Project depth into camera space
	[pixX, pixY] = np.meshgrid(list(range(640)), list(range(480)))
	camX = (pixX - cameraIntrinsics[0, 2]) * inputDepth / cameraIntrinsics[0, 0]
	camY = (pixY - cameraIntrinsics[1, 2]) * inputDepth / cameraIntrinsics[1, 1]
	camZ = inputDepth

	# Only use points with valid depth and within foreground mask
	validDepth = np.logical_and(foregroundMask, camZ != 0)
	xPoints = camX[validDepth]
	yPoints = camY[validDepth]
	zPoints = camZ[validDepth]

	# repackage x, y, z lists into point tuples
	numPoints = len(xPoints)
	inputPoints = [(xPoints[i], yPoints[i], zPoints[i]) for i in range(numPoints)]
	
	# Get foreground point cloud normals
	foregroundPointcloud = pcl.PointCloud()
	foregroundPointcloud.from_list(inputPoints)
	foregroundNormals = calc_surface_normals(foregroundPointcloud)

	# Flip normals to point toward sensor
	sensorCenter = np.zeros(3)
	# did this weird because foregroundNormals has curvature, can't directly np.asarray it
	foregroundPointcloudArray = np.asarray(foregroundPointcloud)
	foregroundNormalsList = []
	for i in range(len(inputPoints)):
		foregroundNormalsList.append(np.asarray(foregroundNormals[i])[0:3]) # ignore curvature value (index 3)
	foregroundNormalsArray = np.asarray(foregroundNormalsList)
	# foregroundNormalsArray = np.asarray([np.asarray(foregroundNormals[i, 0:3]) for i in range(len(inputPoints))])
	# inputPoints = inputPoints.T
	for k in range(len(inputPoints)):
		p1 = sensorCenter - foregroundPointcloudArray[k]
		p2 = foregroundNormalsArray[k]
		angle = np.arctan2(p1.dot(p2.T), np.linalg.norm(np.cross(p1, p2)))
		if angle <= np.pi / 2 and angle >= -np.pi / 2:
			foregroundNormalsArray[k] *= -1

	# Project normals back to image plane
	inputPointsArray = np.asarray(inputPoints)
	pixX = np.around((inputPointsArray[:, 0] * cameraIntrinsics[0, 0]) / (inputPointsArray[:, 2]) + cameraIntrinsics[0, 2]).astype(int)
	pixY = np.around((inputPointsArray[:, 1] * cameraIntrinsics[1, 1]) / (inputPointsArray[:, 2]) + cameraIntrinsics[1, 2]).astype(int)
	# matlab does a weird linear indexing thing here, so just flatten and roll with it
	surfaceNormalsMap = np.zeros_like(inputColor)
	surfaceNormalsMapFlat = np.reshape(surfaceNormalsMap, (-1,)) # this is a little redundant but whatever
	surfaceNormalsMapFlat[np.ravel_multi_index((pixY, pixX, np.zeros_like(pixY).astype(int)), surfaceNormalsMap.shape)] = foregroundNormalsArray[:, 0]
	surfaceNormalsMapFlat[np.ravel_multi_index((pixY, pixX, np.ones_like(pixY).astype(int)), surfaceNormalsMap.shape)] = foregroundNormalsArray[:, 1]	
	surfaceNormalsMapFlat[np.ravel_multi_index((pixY, pixX, 2 * np.ones_like(pixY).astype(int)), surfaceNormalsMap.shape)] = foregroundNormalsArray[:, 2]
	surfaceNormalsMap = np.reshape(surfaceNormalsMapFlat, inputColor.shape) # reshape after that matlab imitation flattish thing

	# Compute standard deviation of local normals
	meanStdNormals = np.mean(window_stdev(surfaceNormalsMap, 25) * np.sqrt((25 ** 2) / (25 ** 2 - 1)), axis=2)
	affordanceMap = 1 - meanStdNormals / np.max(meanStdNormals)
	affordanceMap[validDepth != True] = 0
	print("Finished predict!")
	return affordanceMap, surfaceNormalsMap

# from https://github.com/strawlab/python-pcl/blob/master/examples/sift.py
def calc_surface_normals(cloud):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(0.5)
    cloud_normals = ne.compute()
    return cloud_normals # returns an (N, 4) matrix because logic (last val is curvature I think)
