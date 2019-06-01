The purpose of the project is to identitfy the 3d coordinates of small test image to the large reference 3d reconstructed cloud point.
![procedure](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/Untitled%20Diagram.jpg)
1. VisualSFM and SiftGPU are implemented to run sfm with bundle adjustment to recover sparse cloud point, then CMVS method is used to 
generate dense cloud point (onlly useful for 3d visualization)
The sparse cloud point data is shown in the file bundle.rd.out, but I seperate it into two parts
one for camera information:
a. LiuHong_Camera_index_R_T
format:
<num_cameras> <num_points> [two integers]
<camera1>
<camera2>
   ...
<cameraN>

*****=================================*****************
Camera geometry have the following format:
<f> <k1> <k2>  [the focal length, followed by two radial distortion coefficients]
<R>            [a 3x3 matrix representing the camera rotation]
<t>            [a 3-vector describing the camera translation]
b. LiuHong_Point_Collec
<point1>
<point2>
  ...
<pointM>

*****=================================*****************
Each point entry has the form:

<position> [a 3-vector describing the 3D position of the point]
<color>   [a 3-vector describing the RGB color of the point] I think it is RGB instead of BGR ^_^
<view list> [a list of views the point is visible in]
*****=================================******************
<view list> begin with the length of the list(i.e the number of cameras the point is visible in).
The list is then given as a list of quadruplets <camera> <key> <x> <y>, where <camera> is a camera index,
<key> is the index of the sift keypoint where the point was detected in that camera (for this project, we
are not going to use this parameter), and <x>, <y> are the detected positions of that keypoint. Both indices are 
0-based.
****important: The pixel positions are floating point numbers in a coordinate system where the origin is the
center of the image, the x-axis increase to the right, and the y-axis increases towards the top of the image.
Thus, (-w/2,-h/2) is the lower-left corner of the image, and (w/2,h/2) is the top-right corner (where w and h
are the width and height of the image)
The dense cloud point is too big to uploaded, I named it as point-3D.ply. Basically, it is mesh file, it can be opened
by meshlab. If you want to use:
https://drive.google.com/file/d/1e4iw6MDxaOPivZwCI0aAnaXsN2RTK5fc/view?usp=sharing
Otherwise you can use do CMVS dense reconstruction by using VisualSFM
2. The main file is draw_on_3d_.py. First it calls the test image and run opencv sift detection for all reference image.
Visualize the camera location and oriantation:
![Camera](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/result/camera.JPG)
Visualize the sparse 3d reconstrution:
![Sparse_re](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/result/sparse_reconstruction.JPG)
Visualize the dense 3d reconstruction:
![Dense_Re](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/result/dense_reconstruction.JPG)
All blue points mean camera position

The test image used is:
![test2](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/result/test2/test2.jpg)

3. In order to imporve Keypoint matching efficiency, it is a good idea to reduce the reference image as much as possible.
Region of Interest (ROI) is processed to calculate the bounding box of the bridge by using bothe CNN based segmentation and conventional methods of detection. 
CNN-based approach used 15 images as 10 for training and 5 for testing. VGGNet is adjusted only with fully connected layers at the end
with convolutional layers and added a softmax layer to obtain probability maps for the background and the bridge.
The network was trained for 200 iterations with a batch size of 16 using SGD of learning rate 0.0001 and momentum 0.9

ROI:
![ROI cropping](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/result/ROI.png)

Line detection:
![Line_detection](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/result/line.png)

CNN:
![CNN](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/result/cnn.PNG)

FLANN matching implemented between test image and each reference image with distance filter 0.7.
The matching score are calculated based on the number of matching points, only top 5 matching score reference images are chosen 
to run following part.
![matching_score](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/result/test2/matching_score_for_100_image.png)

4. Remove the 2d outliers from top 5 reference image by calculating mean and std of 2d distance and only using point whose distance is within 1 std. Visualize the matching result.
![matching_result](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/result/test2/Figure_0.png)

5. Run region crop to top 5 reference image:
![region](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/result/test2/Figure_0-1.png)

6. Load projection matrix for those 5 images.

7. Run sift detection and FLANN matching amoung those 5 images to match the feature.

8. Use triangulation method to 3d reconstruct the matching point amount 5 images and apply outlier filter again to filter out the 
outlier from 3d points.

9. Compute mean and std to decide the 3d cube location and width, finall visualize them with dense reconstruction
![final](https://github.com/lipilian/3D_reconstruction_of_bridge/blob/master/result/test2/final_more.JPG)
The red cube indicate the location of test image
