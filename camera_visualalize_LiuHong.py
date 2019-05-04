# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:36:21 2019

@author: liuhong2
"""
#%%
from Camera_load import *
from point_load import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pptk
from open3d import *
import plyfile


#%% calculate the camera center
camera_num,pixel_num,f,R,t = load_camera_data()
center = np.empty([camera_num,3])
vector = np.empty([camera_num,6])
for i in range(camera_num):
    center[i,:] = np.dot(-R[i,:,:].T,t[i,:])
    vector[i,0:3] = np.dot(-R[i,:,:].T,t[i,:])
    vector[i,3:6] = np.dot(R[i,:,:].T,np.array([0,0,1]).reshape(3,))

#%% test the camera location in Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(center[:,0],center[:,1],center[:,2],c='r')
ax.quiver(vector[:,0],vector[:,1],vector[:,2],-vector[:,3],-vector[:,4],-vector[:,5],length = 0.1,arrow_length_ratio = 0.5)
ax.pbaspect = [1.0, 1.0, 1.0]

#%% test the camera location in pptk viewer

#%% load sparse data
def sparse_recon():
    point_position, point_color,point_2dlocation,point_camera = load_point_data()
    xyz = np.double(point_position)
    xyz = np.concatenate((center,xyz))
    camera_color = np.array([[0.,0.,255.],]*camera_num)
    point_color = np.concatenate((camera_color,point_color))
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    pcd.colors = Vector3dVector(point_color/255.)
    draw_geometries([pcd])
    return xyz,point_color
#%% visualize the dense cloud point
def dense_recon():
    plydata = plyfile.PlyData.read('point-3D.ply')
    x = np.double(plydata.elements[0].data['x'])
    x = x.reshape(len(x),1)
    y = np.double(plydata.elements[0].data['y'])
    y = y.reshape(len(y),1)
    z = np.double(plydata.elements[0].data['z'])
    z = z.reshape(len(z),1)
    r= np.double(plydata.elements[0].data['diffuse_red'])
    r = r.reshape(len(r),1)
    g= np.double(plydata.elements[0].data['diffuse_green'])
    g = g.reshape(len(r),1)
    b= np.double(plydata.elements[0].data['diffuse_blue'])
    b = b.reshape(len(r),1)
    xyz = np.concatenate((x,y,z),axis = 1)
    rgb = np.concatenate((r,g,b),axis = 1)
    #camera center recover
    xyz = np.concatenate((center,xyz))
    camera_color = np.array([[0.,0.,255.],]*camera_num)
    rgb = np.concatenate((camera_color,rgb))
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    pcd.colors = Vector3dVector(rgb/255.)
    draw_geometries([pcd])
    return xyz,rgb

#%% 

'''
points = [[0,0,0],[2,0,0],[0,2,0],[2,2,0],
              [0,0,2],[2,0,2],[0,2,2],[2,2,2]]
lines = [[0,1],[0,2],[1,3],[2,3],
             [4,5],[4,6],[5,7],[6,7],
             [0,4],[1,5],[2,6],[3,7]]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = LineSet()
line_set.points = Vector3dVector(points)
line_set.lines = Vector2iVector(lines)
line_set.colors = Vector3dVector(colors)
draw_geometries([line_set,pcd])
'''

