# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:59:56 2019

@author: liuhong2
"""
#%%
import numpy as np
from Camera_load import *
#%%

#%%
def load_point_data():
    camera_num,pixel_num,f,R,t = load_camera_data()
    file = open('LiuHong_Point_Collect.txt','r')
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')
    point_color = np.empty([pixel_num,3]).astype(int)
    point_position = np.empty([pixel_num,3])
    point_camera = []
    point_2dlocation = []
    i = 0
    j = 0
    while i < pixel_num:
        point_position[i,:] = np.double(lines[j].split(' '))
        j += 1
        point_color[i,:] = np.array(lines[j].split(' ')).astype(int)
        j += 1
        
        info = list(np.double(lines[j].split(' ')))
        num = info[0]
        info = info[1:]
        while len(info)/3 < num:
            j+=1
            info = info +  list(np.double(lines[j].split(' ')))
        k = 0
        index_camera = []
        location = []
        while k < len(info):
            index_camera.append(info[k])
            k+=2
            location.append(info[k])
            k+=1
            location.append(info[k])
            k+=1
        point_camera.append(index_camera)
        point_2dlocation.append(location)
        j +=1
        i += 1

    return point_position, point_color,point_2dlocation,point_camera
    
#%%
