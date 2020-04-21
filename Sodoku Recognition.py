
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:01:32 2020

@author: sdien
"""

################################################################
#            Identifying Sudoku Field                          #
################################################################

import os
os.getcwd()
os.chdir("C:\\Users\\sdien\\Documents\\GitHub\\SudokuSolver")

# import libraries
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt


def load_imgdata():
    imgs = []
    path = "data\\sudoku_img\\mixed\\"
    for file in os.listdir(path):
        imgs.append(cv2.imread(os.path.join(path,file)))
    return imgs

# matplotlib summary
def summary(image):
    plt.imshow(image)
    print('Data Type: ', image.dtype)
    print('Dimensions: ', image.shape)
    
    
    
   

############## Image Processing ##############
# load image dataset
imgs = load_imgdata()
summary(imgs[1]) 

img = imgs[1] 

# i) use YUV color space
image_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
image_y = np.zeros(image_yuv.shape[0:2],np.uint8)
image_y[:,:] = image_yuv[:,:,0]

summary(image_y)

# ii)  apply blur
image_blurred = cv2.GaussianBlur(image_y, (5, 5), 0) 

summary(image_blurred)

# iii) use edge detector
edges = cv2.Canny(image_blurred, 30, 150)

# iv) find contours
# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

cv2.drawContours(img,cnts,-1,(255,0,0),3)
summary(img)




# v) find correct contour
for cnt in contours:
    hull = cv2.convexHull(cnt)
    simplified_cnt = cv2.approxPolyDP(hull,0.001*cv2.arcLength(hull,True),True)




edges = cv2.Canny(img, 20, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 2) 
    
summary(img)

# save variable
img_post = img
img = img_post


# Crop sudoku
height, width = img.shape[:2]

#let's get the starting pixel coordinates(top left of cropping rectangle)
start_row, start_col=int(height*.25),int(width*.25)
#let's get ending pixel coordinates(bottom right)
end_row, end_col=int(height*.75),int(width*.75)
#simply use indexing to crop out the rectangle we desire
cropped=img[start_row:end_row, start_col:end_col]
summary(img)
summary(cropped)










# Hough Algorithm
for line in lines[0:1]:       
    # get coordinates of lines
    rho, theta = line[0]
    print("rho ", rho, "theta ",theta)
    a = np.cos(theta)
    b = np.sin(theta)
    print("a: ", a, "b: ", b)
    x0 = a * rho
    y0 = b * rho
    print("x0 ", x0, "y0 ",y0)
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    print("x1 ", x1, "y1 ",y1)
    x2 = int(x0 + 1000 * (-b))
    y2 = int(y0 + 1000 * (a))
    print("x2 ", x2, "y2",y1)
    # plot lines on image  (image, point1, point2, color, line thickness) 
    line = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 5) 
    

cv2.imwrite('houghlines3.jpg',img)

img = cv2.line(img, (400,400), (250, 250), (255,0,0), 5)
summary(img)


# compare Canny vs original



plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()




# see later if that works better:
img = cv2.GaussianBlur(img.copy(), (5, 5), 0) # apply blur
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) 
    # specify adaptive gaussian threshhold: see https://bit.ly/2xN2v3I)
    
img = cv2.bitwise_not(img, img)  # turn image into binary
kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8) 
img = cv2.dilate(img, kernel) # dilate to reduce noise
    
   
plt.imshow(img, cmap = 'gray')
       
       