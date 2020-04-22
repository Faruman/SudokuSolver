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

############## Image Processing ##############
def crop_sudoku(img):
    img_org = img
    # image preprocessing   
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to gray
    plt.imshow(img_gray)
    
    # find largest box from contours
    img_edge = cv2.Canny(img_gray,30,150) # maybe use CANNY instead?
    plt.imshow(img_edge)
    
    _, contours,_ = cv2.findContours(img_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # get contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True) # sort by area
    
    best_cnt = contours[0]

    """# draw rectangel around largest box
    x,y,w,h = cv2.boundingRect(best_cnt)
    cv2.rectangle(img_org,(x,y),(x+w,y+h),(0,255,0),2)
    plt.imshow(img_org) """

    # cut image around largest box  
    x, y, w, h = cv2.boundingRect(best_cnt) # get contour coordinates
    img_out = img_org[y:y+h, x:x+w]
    return img_out


# test: crop all data load data ###########
load_imgdata()
i=0
path = "data\\sudoku_img\\sudoku_crop\\"
for img in imgs: # crop all images
    i += 1
    print(i)
    filename = "image"+str(i)+".jpg"

    img_crop = crop_sudoku(img)
    cv2.imwrite(path+filename, img_crop)
# test end #################################


img = imgs_upload[3]
plt.imshow(img)
plt.imshow(crop_sudoku(img))


# apply CANNY second time for numbers
c = crop_sudoku(img)
img_edge2 = cv2.Canny(c,30,150)
plt.imshow(img_edge2)
