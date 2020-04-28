
##############################################################################
#                      Identifying Sudoku Field                              #
##############################################################################

import os
os.getcwd()
os.chdir("C:\\Users\\sdien\\Documents\\GitHub\\SudokuSolver")

# import libraries

import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf

def load_imgdata():
    global imgs, imgs_upload
    imgs = []
    imgs_upload = []
    path1 = "data\\sudoku_img\\mixed\\"
    path2 = "data\\sudoku_img\\uploads\\"
    for file in os.listdir(path1):
        imgs.append(cv2.imread(os.path.join(path1,file)))
    for file in os.listdir(path2):
        imgs_upload.append(cv2.imread(os.path.join(path2,file)))
    return imgs, imgs_upload

def crop_sudoku(img):
    # Detect edges with "Canny Edge Detector"
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to gray
    img_edges = cv2.Canny(img_gray,30,150) # detects edges in image
    img_closed = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) # smooth lines

    # Find largest contour            #!! maybe approx poly als contour probieren
    _, contours,_ = cv2.findContours(img_closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # get contours
    largest_cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0] # sort by area and take first element

    # Draw largest rectangle
    rect = np.zeros(img.shape[0:2], dtype=np.uint8)
    rect = cv2.drawContours(rect, [largest_cnt], 0, (255), 2)
    
    # Find corners points with "Shi-Tomasi Corner Detector"
    pts = cv2.goodFeaturesToTrack(rect, minDistance = 20 ,maxCorners = 4, qualityLevel=0.01)
    pts = np.int0(pts)
    pts = pts.reshape((4,2))

    # Sort corner points
    s = pts.sum(axis=1)             # x + y
    tl = tuple(pts[np.argmin(s)])   # lower sum; top left
    br = tuple(pts[np.argmax(s)])   # higer sum; bottom right

    diff = np.diff(pts, axis=1)         # x - y
    tr = tuple(pts[np.argmin(diff)])    # lower diff; top right
    bl = tuple(pts[np.argmax(diff)])    # higer diff; bottom left
   
    # Adjust perspective 
    rows = cols = 500
    
    pts1 = np.float32([tl,tr,bl,br])
    pts2 = np.float32([[0,0],[cols,0],[0,rows],[cols, rows]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    cimg = cv2.warpPerspective(img,M,(cols, rows))

    return cimg
    

def crop_numbers(cimg):
    # Split image into 81 numbers
    nums = [] 
    img_height, img_width = cimg.shape[0:2]
    h = int(img_height/9) + 1
    w = int(img_width/9) + 1
    for r in range(0,img_height, h):
        for c in range(0,img_width, w):
            nums.append(cimg[r:r+h , c:c+w])

    # Clean numbers 
    for i in range(len(nums)):  
        n = nums[5]
        n_res = cv2.resize(n, (100, 100) , interpolation = cv2.INTER_CUBIC) # fix size to 100, 100
        n_blur = cv2.medianBlur(n_res, 3)
        n_gray = cv2.cvtColor(n_blur, cv2.COLOR_BGR2GRAY) # convert to gray
        n_bw = cv2.adaptiveThreshold(n_gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2) # convert to binary
        n_bw = cv2.bitwise_not(n_bw)
        n_closed = cv2.morphologyEx(n_bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) # smooth lines
        n_closed = cv2.morphologyEx(n_closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) # smooth lines
    
        plt.imshow(n, cmap = 'gray')
        plt.imshow(n_blur, cmap = 'gray')
        plt.imshow(n_gray, cmap = 'gray')
        plt.imshow(n_bw, cmap = 'gray')
        plt.imshow(n_closed, cmap = 'gray')

        # Center number
        _, contours,_ = cv2.findContours(n_closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # get contours
        largest_cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0] # sort by area and take first element

        # Draw largest rectangle
        grid = np.zeros(n.shape[0:2], dtype=np.uint8)
        len(contours)      
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            grid = cv2.drawContours(grid, [box], -1, (255), 2)
       
        plt.imshow(grid, cmap = 'gray')
    
        nums[i] = n_closed
    
    # Plot Numbers 
    ROW = 9
    COLUMN = 9
    for i in range(ROW*COLUMN): 
        n = nums[i]
        plt.subplot(ROW, COLUMN, i+1)        # subplot with size
        plt.axis('off')
        plt.imshow(n, cmap='gray')  # cmap='gray_r' is for black
        
    return nums
    

    
############## Test on one image##############
load_imgdata()
img = imgs[1]
plt.imshow(img)

cimg = crop_sudoku(img)
plt.imshow(cimg)

numbers = crop_numbers(cimg)
plt.imshow(numbers[6], cmap = 'gray')
############## Test End ######################





# load digit recognition model
ConvModel = tf.keras.models.load_model('model')

# predict digits

    
    
    
    
    #%% Other stuff   
plt.subplot(131), plt.imshow(img), plt.axis('off')
plt.subplot(132), plt.imshow(cimg), plt.axis('off')
plt.subplot(133), plt.imshow(numbers_plot, 'gray_r'), plt.axis('off')
plt.show()
    
    