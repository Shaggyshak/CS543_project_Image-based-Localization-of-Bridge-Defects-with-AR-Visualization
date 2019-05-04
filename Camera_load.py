# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:37:41 2019

@author: liuhong2
"""
#%%
import numpy as np
import pandas as pd
#%% read LiuHong_Camera_index_R_T file
def load_camera_data():
    file = open('LiuHong_Camera_index_R_T.txt','r')
    lines = file.readlines()

    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')

    title = np.double(lines[0].split(' '))
    camera_num = int(title[0])
    pixel_num = int(title[1])
    R = np.empty([camera_num,3,3])
    f = np.empty(camera_num)
    t = np.empty([camera_num,3])
      
    i = 1
    j = 0
    while i < len(lines):
        f[j] = np.double(lines[i].split(' '))[0]
        R[j,0,:] = np.double(lines[i+1].split(' '))
        R[j,1,:] = np.double(lines[i+2].split(' '))
        R[j,2,:] = np.double(lines[i+3].split(' '))
        t[j,:] = np.double(lines[i+4].split(' '))
        j+=1
        i+=5
    return camera_num,pixel_num,f,R,t
#%%
