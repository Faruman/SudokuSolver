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
from imutils import contours



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
    #plt.imshow(img_gray)
    
    # find largest box from contours
    img_edge = cv2.Canny(img_gray,30,150) # maybe use CANNY instead?
    #plt.imshow(img_edge)
    
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

load_imgdata()
img = imgs[1]
plt.imshow(img)

img = crop_sudoku(img)
plt.imshow(crop_sudoku(img))


def crop_numbers(img):
    img_org = img
    
    # i) Load image, grayscale, and adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,57,5)
    plt.imshow(thresh)

    # ii) Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

    # iii) Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

    plt.imshow(thresh)

    # iv) Sort by top to bottom and each row by left to right # WAS MACHT DAS??
    invert = 255 - thresh
    cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
    
    sudoku_rows = []
    row = []
    for (i, c) in enumerate(cnts, 1):
        area = cv2.contourArea(c)
        if area < 50000:
            row.append(c)
            if i % 9 == 0:  
                (cnts, _) = contours.sort_contours(row, method="left-to-right")
                sudoku_rows.append(cnts)
                row = []
    
    # Iterate through each box
    for row in sudoku_rows:
        for c in row:
            mask = np.zeros(img.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, (255,255,255), -1)
            result = cv2.bitwise_and(img, mask)
            result[mask==0] = 255
            cv2.imshow('result', result)
            cv2.waitKey(175)
    
    plt.imshow('thresh', thresh)
    plt.imshow('invert', invert)





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
    
plt.imshow(crop_numbers(img))  

# apply CANNY second time for numbers
c = crop_sudoku(img)
img_edge2 = cv2.Canny(c,30,150)
plt.imshow(img_edge2)
