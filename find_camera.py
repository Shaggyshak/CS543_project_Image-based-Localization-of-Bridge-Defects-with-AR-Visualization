# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:13:09 2019

@author: liuhong2
"""
#%%
import os
import sys
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import heapq
import matplotlib.patches as patches
#%% load model
sift = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) 
flann = cv2.FlannBasedMatcher(index_params,search_params)
num_matches= []


#%% test_image load
def load_test(test_dir):
    test_image = cv2.imread(test_dir)
    test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(test_image,None)
    return test_image,kp1,des1



#%% reference dection

def detect_reference():
    ref_kp = []
    ref_des = []
    reference_dir = './reference_image'
    reference_name = os.listdir(reference_dir)
    i = 0
    for ele in reference_name:
        reference_name[i] = reference_dir + '/' + ele
        i+= 1
    refe_image = np.empty([3598,4797,100])
    for j in range(len(reference_name)):
        a = cv2.imread(reference_name[j])
        a = cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
        kp2,des2 = sift.detectAndCompute(a,None)
        ref_kp.append(kp2)
        ref_des.append(des2)
        print(j)
    return ref_kp,ref_des,reference_name

#%% match between test image and reference image
def match_all(reference_name,ref_kp,ref_des,des1):
    num_matches= []
    for j in range(len(reference_name)):
        kp2 = ref_kp[j]
        des2 = ref_des[j]
        matches = flann.knnMatch(des1,des2,k=2)
        matchesMask = [[0,0] for i in range(len(matches))]
        k = 0
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
                k +=1
        num_matches.append(k)
        print(k)
        print(j)
    return num_matches
#%% visualizae the hist graph
def visual_hist(num_matches):
    plt.figure(1,figsize = (6,3));
    x_pos = np.arange(len(num_matches));
    plt.bar(x_pos,num_matches)
    #pick heightest 5 histgrom
    valid_index = heapq.nlargest(5, range(len(num_matches)), np.asarray(num_matches).take)
    valid_matches = []
    for i in range(len(valid_index)):
        valid_matches.append(num_matches[valid_index[i]])
    plt.bar(valid_index,valid_matches,color = (1.0,0,0,1))
#convert int list to string
    def convert(s): 
        new = "" 
        for x in s: 
            new += str(x)+ " "  
        return new 
    valid_index_print = convert(valid_index)
    plt.title(['heighest matches images ' + valid_index_print])
    plt.xlabel('image name')
    plt.ylabel('matches_point')
    return valid_index

#%% visulize the highest matches picture
def visual_match(valid_index,ref_kp,ref_des,kp1,des1,test_dir,reference_name):
    for j in range(len(valid_index)):
        highest_index = valid_index[j]
        kp2 = ref_kp[highest_index]
        des2 = ref_des[highest_index]
        matches = flann.knnMatch(des1,des2,k=2)
        Mask = [[0,0] for i in range(len(matches))]
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                Mask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = Mask,
                           flags = 0)
        image1 = cv2.imread(test_dir)
        image2 = cv2.imread(reference_name[highest_index])      
        img3 = cv2.drawMatchesKnn(image1,kp1,image2,kp2,matches,None,**draw_params)
        print(j)
        plt.figure(j)
        plt.imshow(img3)
#%% remove outlier
def remove_out(valid_index,ref_kp,ref_des,kp1,des1,test_image):
    matched_point_all_image = []
    q,t = test_image.shape
    for i in range(len(valid_index)):
        index = valid_index[i]
        kp2 = ref_kp[index]
        des2 = ref_des[index]
        matches = flann.knnMatch(des1,des2,k=2)
        matched_point = []
        for j,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                pt = kp2[m.trainIdx].pt
                matched_point.append(pt)
        matched_point = np.asarray(matched_point)
        distance = np.sqrt(np.sum(matched_point**2,axis=1)) 
        mean = np.mean(distance)
        std = np.std(distance)
        low = mean - std
        high = mean + std
        good_point_index = np.where(np.logical_and(distance>= low,distance <= high))
        matched_point_filtered = matched_point[good_point_index]
        matched_point_all_image.append(matched_point_filtered)
    return matched_point_all_image               
#%% get 2d location matches in 2d sparse camera
def camera2d_point_loc(valid_index,point_camera,matched_point,point_2dlocation,point_position):
    good_point = []
    for i in range(len(valid_index)):
        camera = valid_index[i]
        refe_point = matched_point[i]
        x = refe_point[:,0]
        y = refe_point[:,1]
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        j = 0
        while j < len(point_camera):
            try:
                index = point_camera[j].index(camera)
                x1 = point_2dlocation[j][2*index]+4797/2.
                y1 = point_2dlocation[j][2*index+1]+3598/2.
                if (x1>=x_min and x1<=x_max and y1>=y_min and y1<=y_max):
                    good_point.append(point_position[j])
                j += 1
            except:
                j += 1
            
    return good_point
#%% get repeated point from good point
def compute_center(good_point):
    good_point = np.asarray(good_point)
    
    return np.mean(good_point,axis = 0), np.std(good_point,axis = 0)
    
#%% projection matrix 
def get_projections(valid_index):
    k = 0
    txt_name ='000000'
    proj_m = np.zeros((100, 3, 4))
    for i in range(100):
        if(i < 10):
            name = txt_name + '0'+ str(k) + '.txt'
        else:
            name = txt_name + str(k) + '.txt'
        file = open('./txt/' + name, 'r')
        lines = file.readlines()[1:] #To skip 1 line
        mat = []
        for j in range(0, len(lines),3):
            for z in range(3):
                mat.append(list(map(float,lines[j+z].split())))
        proj_m[k] = mat
        k +=1
    new_proj = []
    for i in range(len(valid_index)):
        new_proj.append(proj_m[valid_index[i]])
    return new_proj 
#%% calculate the rectangle region in each image
def rect_cal(matched_point):
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    for i in range(len(matched_point)):
        point = matched_point[i]
        x_min.append(np.min(point[:,0]))
        y_min.append(np.min(point[:,1]))
        x_max.append(np.max(point[:,0]))
        y_max.append(np.max(point[:,1]))
    return x_min,x_max,y_min,y_max
#%% visualize each image with rectangle
def visual_rect(x_min,x_max,y_min,y_max,valid_index,reference_name):
    for i in range(len(valid_index)):
        left_corner = (x_min[i],y_min[i])
        width = x_max[i] - x_min[i]
        height = y_max[i] - y_min[i]
        fig = plt.figure(i)
        ax = fig.add_subplot(111)
        img = cv2.imread(reference_name[valid_index[i]])
        ax.imshow(img)
        rect = patches.Rectangle(left_corner,width,height,fill=False,edgecolor='r')
        ax.add_patch(rect)
#%% sift detection within region
def d3_point_calculate(ref_des,ref_kp,x_min,x_max,y_min,y_max,valid_index):
    kp_all = []
    des_all = []
    for i in range(len(valid_index)):
        des = ref_des[valid_index[i]]
        kp = ref_kp[valid_index[i]]
        new_kp = []
        new_des = []
        for j in range(len(kp)):
            if kp[j].pt[0] > x_min[i] and kp[j].pt[0] < x_max[i] and kp[j].pt[1] > y_min[i] and kp[j].pt[1] < y_max[i]:
                new_kp.append(kp[j])
                new_des.append(des[j])
        new_kp = np.asarray(new_kp)
        new_des = np.asarray(new_des)
        kp_all.append(new_kp)
        des_all.append(new_des)
    return kp_all,des_all
#%% matching between region
def region_detect(kp_all,des_all,valid_index):
    result = []
    for i in range(len(valid_index)-1):
        for j in range(i+1,len(valid_index)):
            kp1 = kp_all[i]
            kp2 = kp_all[j]
            des1 = des_all[i]
            des2 = des_all[j]
            cam1 = valid_index[i]
            cam2 = valid_index[j]
            matches = flann.knnMatch(des1,des2,k=2)
            pt1 = []
            pt2 = []
            for s,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    pt1.append(kp1[m.queryIdx].pt)
                    pt2.append(kp2[m.trainIdx].pt)
            result.append([pt1,pt2,i,j])
    return result   
#%% use trigulation method to get 3d coordinate
def tri_method(region_match,proj):
    points = []
    for i in range(len(region_match)):
        data = region_match[i]
        cam1 = data[2]
        cam2 = data[3]
        p1 = proj[cam1]
        p2 = proj[cam2]
        point2d1 = np.asarray(data[0]).T
        point2d2 = np.asarray(data[1]).T
        d3_point = cv2.triangulatePoints(p1,p2,point2d1,point2d2)
        d3_point = d3_point[0:3,:]/d3_point[3,:]
        points.append(d3_point.T)
    d3_point = np.concatenate((points[0],points[1],points[2],points[3],points[4]),axis = 0)
        
    return d3_point
#%% remove out for final result
def remove_out2(d3_point):
    
    distance = np.sqrt(np.sum(d3_point**2,axis=1))
    mean = np.mean(distance)
    std = np.std(distance)
    low = mean - std
    high = mean + std
    good_point_index = np.where(np.logical_and(distance>= low,distance <= high))
    d3_point = d3_point[good_point_index]

    return d3_point

'''  
     mean = np.mean(d3_point,axis=0)
    std = np.std(d3_point,axis=0)
    m,n = d3_point.shape
    new_point = []        
    for i in range(m):
        count = 0
        for j in range(n):
            if d3_point[i,j] > mean[j] + std[j]:
                count +=1
            if d3_point[i,j] < mean[j] - std[j]:
                count +=1
        if count == 0:
            new_point.append(d3_point[i,:])    
'''          
           
        









