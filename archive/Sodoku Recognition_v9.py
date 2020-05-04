
##############################################################################
#                      Identifying Sudoku Field                              #
##############################################################################

# Setup
import os
os.getcwd()
os.chdir("C:\\Users\\sdien\\Documents\\GitHub\\SudokuSolver")

import numpy as np
import cv2
from matplotlib import pyplot as plt


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
    _, contours,_ = cv2.findContours(img_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # get contours
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
        n_res = cv2.resize(num, (num_height, num_width) , interpolation = cv2.INTER_CUBIC) # fix size to 50,50
        n_blur = cv2.medianBlur(n_res, 3)
        n_gray = cv2.cvtColor(n_blur, cv2.COLOR_BGR2GRAY) # convert to gray

        nums_processed.append(n_gray)
        
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

img = imgs_upload[2]     # somewhat working example: imgs[1], img_uplaod[1,3]
plt.imshow(img)


cimg = extract_sudoku(img)
plt.imshow(cimg)


numbers_list = extract_numbers(cimg)
numbers_plot = list81_to_image(numbers_list)
plt.imshow(numbers_plot, 'gray')



   
    
         
#%% Other Stuff   
    
            
        """# Filter based on area
        areas = stats[:,-1] # extract area from stats
        areas = np.where(areas == np.max(areas), 0, areas)  # set largest component (= background) to 0
        areas = np.where(areas < 0.01 * np.sum(areas), 0, areas) # set components below 1% (= noise) to 0
        idx = np.nonzero(areas)"""
    
    