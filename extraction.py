

import numpy as np
class dataLoader():
	def __init__(self, root):
		self.root = root


	def load_txt(self, txt_file):
		file = open(self.root+txt_file, 'r')
		lines = file.readlines()[18:] #To skip 18 lines.

		for line in lines:
		    if('H:' in line):
		        del lines[lines.index(line)]
		self.lines = lines



	### focal length array ###
	def get_focal_len(self):
		focal_len = np.zeros((100,1))
		n = len(self.lines)
		k = 0
		for i in range(1,n,13):
		    focal_len[k] = float(self.lines[i])
		    k+=1

		return focal_len


	### Principal point array ###

	def get_principal_points(self):

		principal_point = np.zeros((100,2))
		n = len(self.lines)

		k = 0
		for i in range(2,n,13):
		    line = self.lines[i].split()
		    principal_point[k] = list(map(float, line))
		    k+=1

		return principal_point


	### translation array ###
	def get_translation(self):
		trans = np.zeros((100,3))
		n = len(self.lines)

		k = 0
		for i in range(3,n,13):
		    line = self.lines[i].split()
		    trans[k] = list(map(float, line))
		    k+=1

		return trans


	### Camera Position array ###
	def get_camera_pos(self):
		cam_pos = np.zeros((100,3))
		n = len(self.lines)

		k = 0
		for i in range(4,n,13):
		    line = self.lines[i].split()
		    cam_pos[k] = list(map(float, line))
		    k+=1

		return cam_pos


	### Rotation matrix array ###
	def get_rotation(self):
		rot = np.zeros((100,3,3))
		n = len(self.lines)

		k = 0
		for i in range(7,n,13):
		    mat = []
		    for j in range(3):
		        line = self.lines[i+j].split()
		        line = list(map(float, line))
		        mat.append(line)
		    rot[k] = mat 
		    k+=1

		return rot

	def get_projections(self):
		n = len(self.lines)

		k = 0
		txt_name ='000000'
		proj_m = np.zeros((100, 3, 4))
		for i in range(100):
		    if(i < 10):
		        name = txt_name + '0'+ str(k) + '.txt'
		    else:
		        name = txt_name + str(k) + '.txt'
		    file = open(self.root + '/txt/' + name, 'r')
		    lines = file.readlines()[1:] #To skip 1 line
		    mat = []
		    for j in range(0, len(lines),3):
		        for z in range(3):
		            mat.append(list(map(float,lines[j+z].split())))

		    proj_m[k] = mat
		    k += 1
		return proj_m

#example of how it works


root = '/Users/Majdi/Documents/CS543/cs543-project/' # your directory to txt_files
txt = 'cameras_v2.txt'


data = dataLoader(root)
data.load_txt(txt)
proj = data.get_projections()





