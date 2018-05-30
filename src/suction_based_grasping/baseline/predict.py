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
# function [affordance_map,surface_normals_map] = predict(input_color,input_depth,background_color,background_depth,camera_intrinsics)
# Input:
#   input_color         - 480x640x3 float array of RGB color values
#   input_depth         - 480x640 float array of depth values in meters
#   background_color    - 480x640x3 float array of RGB color values 
#   background_depth    - 480x640 float array of depth values in meters
#   camera_intrinsics   - 3x3 camera intrinsics matrix
# Output:
#   affordance_map      - 480x640 float array of affordance values in range [0,1]
#   surface_normals_map  - 480x640x3 float array of surface normals in camera coordinates (meters)

def predict(input_color, input_depth, background_color, background_depth, camera_intrinsics):
	print("Starting predict...")
	# Scale color images between [0, 1]
	# input_color /= 255.
	# background_color /= 255.
	
	# Do background subtraction to get foreground mask
	foreground_mask_color = (np.sum(np.abs(input_color - background_color) < 0.3, axis=2) != 3) # mask pixels similar to background
	foreground_mask_depth = np.logical_and(background_depth != 0, np.abs(input_depth - background_depth) > 0.02)
	foreground_mask = np.logical_or(foreground_mask_color, foreground_mask_depth)

	# show masked image
	# plt.figure()
	# showImg = input_color
	# showImg[foreground_mask == False] = 0
	# plt.imshow(showImg)
	# plt.axis('off')

	# Project depth into camera space
	[pix_x, pix_y] = np.meshgrid(list(range(640)), list(range(480)))
	cam_x = (pix_x - camera_intrinsics[0, 2]) * input_depth / camera_intrinsics[0, 0]
	cam_y = (pix_y - camera_intrinsics[1, 2]) * input_depth / camera_intrinsics[1, 1]
	cam_z = input_depth

	# Only use points with valid depth and within foreground mask
	valid_depth = np.logical_and(foreground_mask, cam_z != 0)
	x_points = cam_x[valid_depth]
	y_points = cam_y[valid_depth]
	z_points = cam_z[valid_depth]

	# repackage x, y, z lists into point tuples
	num_points = len(x_points)
	input_points = [(x_points[i], y_points[i], z_points[i]) for i in range(num_points)]
	
	# Get foreground point cloud normals
	foreground_point_cloud = pcl.PointCloud()
	foreground_point_cloud.from_list(input_points)
	foreground_normals = calc_surface_normals(foreground_point_cloud)

	# Flip normals to point toward sensor
	sensor_center = np.zeros(3)
	# did this weird because foreground_normals has curvature, can't directly np.asarray it
	foreground_point_cloud_array = np.asarray(foreground_point_cloud)
	foreground_normals_list = []
	for i in range(len(input_points)):
		foreground_normals_list.append(np.asarray(foreground_normals[i])[0:3]) # ignore curvature value (index 3)
	foreground_normals_array = np.asarray(foreground_normals_list)
	for k in range(len(input_points)):
		p1 = sensor_center - foreground_point_cloud_array[k]
		p2 = foreground_normals_array[k]
		angle = np.arctan2(p1.dot(p2.T), np.linalg.norm(np.cross(p1, p2)))
		if angle <= np.pi / 2 and angle >= -np.pi / 2:
			foreground_normals_array[k] *= -1

	# Project normals back to image plane
	input_points_array = np.asarray(input_points)
	pix_x = np.around((input_points_array[:, 0] * camera_intrinsics[0, 0]) / (input_points_array[:, 2]) + camera_intrinsics[0, 2]).astype(int)
	pix_y = np.around((input_points_array[:, 1] * camera_intrinsics[1, 1]) / (input_points_array[:, 2]) + camera_intrinsics[1, 2]).astype(int)
	# matlab does a weird linear indexing thing here, so just flatten and roll with it
	surface_normals_map = np.zeros_like(input_color)
	surface_normals_map_flat = np.reshape(surface_normals_map, (-1,)) # this is a little redundant but whatever
	surface_normals_map_flat[np.ravel_multi_index((pix_y, pix_x, np.zeros_like(pix_y).astype(int)), surface_normals_map.shape)] = foreground_normals_array[:, 0]
	surface_normals_map_flat[np.ravel_multi_index((pix_y, pix_x, np.ones_like(pix_y).astype(int)), surface_normals_map.shape)] = foreground_normals_array[:, 1]	
	surface_normals_map_flat[np.ravel_multi_index((pix_y, pix_x, 2 * np.ones_like(pix_y).astype(int)), surface_normals_map.shape)] = foreground_normals_array[:, 2]
	surface_normals_map = np.reshape(surface_normals_map_flat, input_color.shape) # reshape after that matlab imitation flattish thing

	# Compute standard deviation of local normals
	mean_std_normals = np.mean(window_stdev(surface_normals_map, 25) * np.sqrt((25 ** 2) / (25 ** 2 - 1)), axis=2)
	affordance_map = 1 - mean_std_normals / np.max(mean_std_normals)
	affordance_map[valid_depth != True] = 0
	print("Finished predict!")
	return affordance_map, surface_normals_map

# from https://github.com/strawlab/python-pcl/blob/master/examples/sift.py
def calc_surface_normals(cloud):
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(0.5)
    cloud_normals = ne.compute()
    return cloud_normals # returns an (N, 4) matrix because logic (last val is curvature I think)
