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
def crop_sudoku(img):
    img_org = img
    # image preprocessing   
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to gray

    # find largest box from contours
    img_edge = cv2.Canny(img_gray,50,200) # maybe use CANNY instead?
    plt.imshow(img_edge)
    
    minLineLength = 100
    maxLineGap = 10
    
    lines = cv2.HoughLinesP(img_edge,1,np.pi/180,100,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    
    plt.imshow(img)
    
    cv2.imwrite('houghlines5.jpg',img)
        
        
    
    
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

imgs()


# test: crop all data load data
imgs = load_imgdata()
i=0
path = "data\\sudoku_img\\sudoku_crop\\"
for img in imgs:
    i += 1
    print(i)
    filename = "image"+str(i)+".jpg"

    img_crop = crop_sudoku(img)
    cv2.imwrite(path+filename, img_crop)
    



