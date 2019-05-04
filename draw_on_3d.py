# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:02:04 2019

@author: liuhong2
"""
#%%
%autoreload 2
import os
import sys
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import heapq
from Camera_load import *
from point_load import *
from camera_visualalize_LiuHong import *
import find_camera 
from mpl_toolkits.mplot3d import Axes3D
import pptk
from open3d import *
import plyfile
import matplotlib.patches as patches
#%% test image dir
test_dir = './cropped_test_image/test2.jpg'
#%% preload sift of reference if not exist
try:
    ref_kp[0]
except:
    print('No preloaded model, sift detecting the reference image now............')
    ref_kp, ref_des, reference_name = find_camera.detect_reference() # get the reference kep point and descrption
    print('Finished the sift detection of the reference image')
#%% load test image data, just 1 image
print('load the test image.........')
test_image,kp1,des1 = find_camera.load_test(test_dir)
print('Finished load the test image')
#%% preview the reference 3d data
print('previewing the sparse reconstruction data');sparse_recon()
print('previewing the reference data.....');xyz,rgb = dense_recon()
#%% try match the test image and reference image and visualize as histogram
print('matching the test image with reference');num_matches = find_camera.match_all(reference_name,ref_kp,ref_des,des1)
print('Finished matching the test image with reference');valid_index = find_camera.visual_hist(num_matches)
print('Drawing highest score matching');find_camera.visual_match(valid_index,ref_kp,ref_des,kp1,des1,test_dir,reference_name)
#%% remove the out lier from the matching data
print('Remove outlier and get matched point in all valid index image');matched_point = find_camera.remove_out(valid_index,ref_kp,ref_des,kp1,des1,test_image)
print('Load the sparse reconstruction information');point_position, point_color,point_2dlocation,point_camera = load_point_data()
print('Load the projection matrix');proj = get_projections(valid_index)
print('Calculate test area in each image');x_min,x_max,y_min,y_max = rect_cal(matched_point)
print('visulize the test region');visual_rect(x_min,x_max,y_min,y_max,valid_index,reference_name)
print('sift detection with in region');kp_all,des_all = d3_point_calculate(ref_des,ref_kp,x_min,x_max,y_min,y_max,valid_index)
print('flann matching within region');region_match = region_detect(kp_all,des_all,valid_index)

#%% triangulate the point
print('Get 3D point');d3_point = tri_method(region_match,proj)
print('filter 3D point');d3_point = remove_out2(d3_point)

#%% draw the box on 3d point cloud
def draw_box(d3_point,xyz,rgb):
    mean = np.mean(d3_point,axis =0)
    minv = np.min(d3_point,axis = 0)
    maxv = np.max(d3_point,axis = 0)
    mean = (maxv+minv)/2
    std = np.mean(np.std(d3_point,axis =0))
    sigma = 10*std
    x = mean[0]-sigma
    y = mean[1]-sigma
    z = mean[2]-sigma
    points = [[x,y,z],[x+sigma,y,z],[x,y+sigma,z],[x+sigma,y+sigma,z],
              [x,y,z+sigma],[x+sigma,y,z+sigma],[x,y+sigma,z+sigma],[x+sigma,y+sigma,z+sigma]]
    lines = [[0,1],[0,2],[1,3],[2,3],
             [4,5],[4,6],[5,7],[6,7],
             [0,4],[1,5],[2,6],[3,7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = LineSet()
    line_set.points = Vector3dVector(points)
    line_set.lines = Vector2iVector(lines)
    line_set.colors = Vector3dVector(colors)
    re = PointCloud()
    re.points = Vector3dVector(xyz)
    re.colors = Vector3dVector(rgb/255.)
    draw_geometries([line_set,re])
print('draw the final graph for visual');draw_box(d3_point,xyz,rgb)
#print('Getting good 3d point matching the test image');good_point = find_camera.camera2d_point_loc(valid_index,point_camera,matched_point,point_2dlocation,point_position)
#print('Calculate the mean location of the good point');center,s = compute_center(good_point)


