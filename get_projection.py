#edited by majdi
import numpy as np

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
         

