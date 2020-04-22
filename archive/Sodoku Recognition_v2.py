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

############## Image Processing ##############

# i) load data
imgs = load_imgdata()
img = imgs[1] 
plt.imshow(img)


# ii)  apply gray scale + further transformations, blur, dilate
def mask_sudoku(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to gray
    mask = np.zeros((gray.shape),np.uint8)      # initialize mask
    
    # apply Morphological Transformations: "Closing"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10,10))
    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE, kernel)
    
    # final normalization
    div = np.float32(gray)/(close)
    res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)
    plt.imshow(res2)
        
    # find contours
    thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
    plt.imshow(thresh)
    
    _, contours,_ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    best_cnt = contours[0]
        
    cv2.drawContours(mask,[best_cnt],0,255,-1)
    cv2.drawContours(mask,[best_cnt],0,0,2)
    
    res = cv2.bitwise_and(res,mask)
    plt.imshow(res)
    
    x,y,w,h = cv2.boundingRect(best_cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    plt.imshow(img)    
    
    

    
    x, y, w, h = cv2.boundingRect(best_cnt) # get contour coordinates
    res = res[y:y+h, x:x+w]
    plt.imshow(res)



