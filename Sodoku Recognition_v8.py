
##############################################################################
#                      Identifying Sudoku Field                              #
##############################################################################

#%% Setup 
import os
os.getcwd()
os.chdir("D:\Programming\Python\SudokuSolver")

import numpy as np
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf

#%% Define functions

def load_imgdata():
    global imgs, imgs_upload, imgs_test
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
    # Find largest contour
    contours,_ = cv2.findContours(img_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # get contours
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
    """##### SELECT EXAMPMLE ######
    num = nums[52]
    plt.imshow(num)
    """
    # Clean numbers 
    nums_processed = []   
    num_height = num_width = 100
    for num in nums:  
        # Convert to grey image
        n_res = cv2.resize(num, (num_height, num_width) , interpolation = cv2.INTER_CUBIC) # fix size to 100, 100
        n_blur = cv2.medianBlur(n_res, 3)
        n_gray = cv2.cvtColor(n_blur, cv2.COLOR_BGR2GRAY) # convert to gray

        # Find best threshold, that lowers number of central components 
        centerpieces = []  # number of centerpieces
        center = np.array([num_height/2, num_width/2]) # define center of image   
        r = 0.35 * num_height
        
        for i in range(2,20):
            # Convert to binary depending on threshold i
            n_bw = cv2.adaptiveThreshold(n_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, i)
            #plt.imshow(n_bw, 'gray')
            # Mask outside frame (= gridlines)
            x = y = int(num_height * 0.05)  
            w = h = int(num_height * 0.9)
            
            mask = np.zeros(n_bw.shape[:2],np.uint8)
            mask[y:y+h,x:x+w] = 255 
        
            n_masked = cv2.bitwise_and(n_bw, n_bw, mask = mask)
            #plt.imshow(n_masked, 'gray')
            # Get all connected components
            output = cv2.connectedComponentsWithStats(n_masked, cv2.CV_32S, connectivity = 8)
            centroids = output[3]
            dist = [*map(lambda pt: np.linalg.norm(pt - center) < r, centroids)] # illustration: cv2.circle(n_bw, (50,50), 35, 225)
            score = sum(dist)
            centerpieces.append(score)
            
            if len(centerpieces) > 3: # break if pices are increasing again
               if all(score > (centerpieces[-4:-1])) : break
             
        # Apply best threshold; get connected components 
        thr = np.argmin(centerpieces) + 2 # convert index to threshold; range started from 2
        n_bw = cv2.adaptiveThreshold(n_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, thr) 

        output = cv2.connectedComponentsWithStats(n_bw, cv2.CV_32S, connectivity = 8)
        labels     = output[1] 
        centroids  = output[3]

        # Filter based on centroid distance to center
        dist = [*map(lambda pt: np.linalg.norm(pt - center) < r, centroids)] # illustration: cv2.circle(n_bw, (50,50), 35, 225)
        dist[0] = False    #dist[0] is backgraud
        idx = np.where(dist)[0]
        
        # Print components
        num_out = np.zeros(labels.shape)
        num_out = np.where(np.isin(labels, idx), 255, 0)
        
        # Smooth lines with Morphological Tansformations
        num_out = np.uint8(num_out)
        num_out = cv2.morphologyEx(num_out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) # smooth lines
        
        nums_processed.append(num_out)
        #plt.imshow(num_out, 'gray')
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

img = imgs_upload[1]     # somewhat working example: imgs[1], img_uplaod[1,3]
plt.imshow(img)
plt.show()

cimg = extract_sudoku(img)
plt.imshow(cimg)
plt.show()

numbers_list = extract_numbers(cimg)
numbers_plot = list81_to_image(numbers_list)
plt.imshow(numbers_plot, 'gray_r')
plt.show()

#### TODO: Filter grid somehow? maybe dont plot if points in stats are extreme like in cornor??
#### TODO: choosing optimal filter still looks spagetti




#%% Classify numbers

# load digit recognition model
ConvModel = tf.keras.models.load_model('model')

# predict digits

    
    
         
#%% Other Stuff   
    
            
"""# Filter based on area
areas = stats[:,-1] # extract area from stats
areas = np.where(areas == np.max(areas), 0, areas)  # set largest component (= background) to 0
areas = np.where(areas < 0.01 * np.sum(areas), 0, areas) # set components below 1% (= noise) to 0
idx = np.nonzero(areas)"""
    
    