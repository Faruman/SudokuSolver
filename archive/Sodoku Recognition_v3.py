################################################################
#            Identifying Sudoku Field                          #
################################################################

import os
os.getcwd()
os.chdir("C:\\Users\\sdien\\Documents\\GitHub\\SudokuSolver")

# import libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt


def load_imgdata():
    imgs = []
    path = "data\\sudoku_img\\mixed\\"
    for file in os.listdir(path):
        imgs.append(cv2.imread(os.path.join(path,file)))
    return imgs  

#load test data
imgs = load_imgdata()
img = imgs[22]
plt.imshow(img)

############## Image Processing ##############
def crop_sudoku(img):
    # image preprocessing
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to gray
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10)) # setup kernel
    img_close = cv2.morphologyEx(img_gray,cv2.MORPH_CLOSE, kernel)     # apply "Closing"

    img_div = np.float32(img_gray)/(img_close)                             
    img_final = np.uint8(cv2.normalize(img_div,img_div,0,255,cv2.NORM_MINMAX)) # min_max normalization
    img_final_BGR = cv2.cvtColor(img_final,cv2.COLOR_GRAY2BGR)  # convert back to BGR
   
    plt.imshow(img_final)   
    # find largest box from contours
    img_thresh = cv2.adaptiveThreshold(img_final,255,0,1,19,2) # maybe use CANNY instead?
    _, contours,_ = cv2.findContours(img_thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # get contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True) # sort by area
    
    best_cnt = contours[0]

    # cut image around largest box  
    x, y, w, h = cv2.boundingRect(best_cnt) # get contour coordinates
    img_final_BGR_crop = img_final_BGR[y:y+h, x:x+w]
    return img_final_BGR_crop

    # draw rectangel around largest box
    x,y,w,h = cv2.boundingRect(best_cnt)
    cv2.rectangle(img_final_BGR,(x,y),(x+w,y+h),(0,255,0),2)
    plt.imshow(img_final_BGR) 
    


# load data
imgs = load_imgdata()
i=0
for img in imgs:
    print(i)
    img_crop = crop_sudoku(img)
    plt.imshow(img_crop)
    i += 1
    


