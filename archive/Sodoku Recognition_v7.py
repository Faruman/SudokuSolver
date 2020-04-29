
##############################################################################
#                      Identifying Sudoku Field                              #
##############################################################################

#%% Setup 
import os
os.getcwd()
os.chdir("C:\\Users\\sdien\\Documents\\GitHub\\SudokuSolver")

import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf

#%% Define functions

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

def extract_sudoku(img):
    # Detect edges with "Canny Edge Detector"
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to gray
    img_edges = cv2.Canny(img_gray,30,150) # detects edges in image
    n_moph = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) # smooth lines

    # Find largest contour
    _, contours,_ = cv2.findContours(n_moph,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # get contours
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
    

def extract_numbers(cimg):
    # Split image into 81 cells with numbers 
    nums = [] 
    img_height, img_width = cimg.shape[0:2]
    h = int(img_height/9) + 1
    w = int(img_width/9) + 1
    for r in range(0,img_height, h):
        for c in range(0,img_width, w):
            nums.append(cimg[r:r+h , c:c+w])
    
    # Clean numbers 
    nums_processed = []   
    num_height = num_width = 100
    for num in nums:  
        # Convert to binary image
        n_res = cv2.resize(num, (num_height, num_width) , interpolation = cv2.INTER_CUBIC) # fix size to 100, 100
        n_blur = cv2.medianBlur(n_res, 3)
        n_gray = cv2.cvtColor(n_blur, cv2.COLOR_BGR2GRAY) # convert to gray
        n_bw = cv2.adaptiveThreshold(n_gray,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2) # convert to binary
        n_bw = cv2.bitwise_not(n_bw)
        
        # Mask gridlines
        x = y = int(num_height * 0.05)                      ####### TODO: make thinner maybe
        w = h = int(num_height * 0.9)
        
        mask = np.zeros(n_bw.shape[:2],np.uint8)
        mask[y:y+h,x:x+w] = 255 
        
        n_masked = cv2.bitwise_and(n_bw, n_bw, mask = mask)
        
        # Apply morphological transformations
        n_moph = cv2.morphologyEx(n_masked, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) # smooth lines
        n_moph = cv2.morphologyEx(n_moph, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) # smooth lines
        
        # Get all connected pixels of binary image
        output = cv2.connectedComponentsWithStats(n_moph, cv2.CV_32S, connectivity = 8)
        labels     = output[1] 
        stats      = output[2]
        centroids  = output[3]
        areas = stats[:,-1] # extract area from stats
         
        # Select only large areas
        if (len(areas) > 1):
            areas = np.where(areas == np.max(areas), 0, areas) # set largest component (= background) to 0
        areas = np.where(areas < np.mean(areas), 0, areas) # set components below mean (= noise) to 0
        areas_idx = np.nonzero(areas) # get index of leftover areas (= number or gridline)
        
        # From those, select the one that is closest to center (= number)
        center = np.array([num_height/2, num_width/2])                              ###### TODO: do check if centroids list is empty
        center_centroid = min(centroids[areas_idx], key=lambda pt: np.linalg.norm(pt - center)) 
        num_label = np.where(centroids == center_centroid)[0][0] # row index 
        
        # Save largest component
        num_out = np.zeros(labels.shape)
        num_out[labels == num_label] = 255        
        nums_processed.append(num_out)
        
    return nums_processed

def list81_to_image(list81):
    h, w = list81[0].shape[0:2]
    plot_height, plot_width = np.multiply(list81[0].shape[0:2], 9)
    
    plot = np.zeros((plot_height, plot_width), np.uint8)

    # overwrite plot with images from list    
    i = 0
    for r in range(0, plot_height, h):
        for c in range(0,plot_width, w):
            (plot[r:r+h , c:c+w]) = list81[i]
            i+=1

    return plot
    
        
#%% TEST: on one image
      
load_imgdata()

img = imgs[1]       # Working example: imgs[1] 
plt.imshow(img)


cimg = extract_sudoku(img)
numbers_list = extract_numbers(cimg)
numbers_plot = list81_to_image(numbers_list)

plt.imshow(img)
plt.imshow(cimg)
plt.imshow(numbers_plot, 'gray_r')





#%% Classify numbers

# load digit recognition model
ConvModel = tf.keras.models.load_model('model')

# predict digits

    
    
    
    