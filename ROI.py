# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('test3.jpg',0)
edges = cv.Canny(img,150,250)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
img2 = img.copy()
Final_ROI = img.copy()
minLengthHOR = 500
maxLineHOR = 30

lines = cv.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 300 ,minLineLength = minLengthHOR ,maxLineGap = maxLineHOR)

X_1 =  np.zeros(lines.shape[0])
Y_1 =  np.zeros(lines.shape[0])
X_2 =  np.zeros(lines.shape[0])
Y_2 =  np.zeros(lines.shape[0])

N = lines.shape[0]
for i in range(N):
    x1 = lines[i][0][0]
    y1 = lines[i][0][1]    
    x2 = lines[i][0][2]
    y2 = lines[i][0][3]
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),10)
print(lines.shape)

print(lines)
plt.figure(),plt.imshow(img),plt.title('Lines_Horizonal'),plt.axis('off')
plt.show()

X_1 = min(np.amin(lines, axis=0)[0,0],np.amax(lines, axis=0)[0,2]) 
Y_1 = min(np.amin(lines, axis=0)[0,1],np.amin(lines, axis=0)[0,3])
X_2 = max(np.amax(lines, axis=0)[0,2],np.amax(lines, axis=0)[0,0])
Y_2 = max(np.amax(lines, axis=0)[0,3],np.amax(lines, axis=0)[0,1])

print(X_1)
print(Y_1)
print(X_2)
print(Y_2)


#if Y_1>Y_2 :
    #temp = Y_2
    #Y_2 = Y_1
    #Y_1 = temp
#if X_1>X_2 :
    #temp = X_2
    #X_2 = X_1
    #X_1 = temp   
cv.line(img,(X_1,Y_1),(X_2,Y_2),(0,0,255),10)
plt.figure(),plt.imshow(img),plt.title('Final_Line'),plt.axis('off')
plt.show()
#plt.figure(),plt.imshow(roi),plt.title('roi'),plt.axis('off')
#plt.show()

###Piers of the bridge part 
#-------------------------------------------------------------------
edgesv = cv.Canny(img2,70,200)
minLengthVER = 500
maxLineVER = 60
temp_img2 = img2.copy()
linesv = cv.HoughLinesP(edgesv,rho = 1,theta = 1*np.pi/180,threshold = 300 ,minLineLength = minLengthVER ,maxLineGap = maxLineVER)
X_1V =  np.zeros(linesv.shape[0])
Y_1V =  np.zeros(linesv.shape[0])
X_2V =  np.zeros(linesv.shape[0])
Y_2V =  np.zeros(linesv.shape[0])

N = linesv.shape[0]
templinesv = np.zeros_like(linesv)
for i in range(N):
    x1v = linesv[i][0][0]
    y1v = linesv[i][0][1]    
    x2v = linesv[i][0][2]
    y2v = linesv[i][0][3]
    if (abs(x1v-x2v))/(((y1v-y2v)**2+(x1v-x2v)**2)**0.5) < 0.5 :
        cv.line(img2,(x1v,y1v),(x2v,y2v),(0,0,255),10)
        templinesv[i][0][0] = x1v
        templinesv[i][0][1] = y1v    
        templinesv[i][0][2] = x2v
        templinesv[i][0][3] = y2v



plt.figure(),plt.imshow(img2),plt.title('lines_Vertical'),plt.axis('off')
plt.show()

print(templinesv)
Y_1V = np.amax(templinesv, axis=0)[0,1]
Y_2V = np.amax(templinesv, axis=0)[0,3]
Y_V = max(Y_1V,Y_2V)
print(Y_1V)
print(Y_2V)
print(Y_V)

if Y_V > Y_2 :
    Y_2 = Y_V

 
roi_final = Final_ROI[Y_1:Y_2, X_1:X_2]
plt.figure(),plt.imshow(roi_final),plt.title('Final_ROI'),plt.axis('off')
plt.show()
