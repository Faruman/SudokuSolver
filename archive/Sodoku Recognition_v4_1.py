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
import math


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

def get_contours(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # convert to gray
    img_edge = cv2.Canny(img_gray,30,150) # find largest box from contours
    
    _, cnts,_ = cv2.findContours(img_edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # get contours
    return cnts

def crop_sudoku(img):
    img_org =  np.copy(img)
 
    contours = get_contours(img)
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

def crop_numbers(cimg): # cropped image (cimg)
    cells = [] 
    
    # plit image into 81 pieces
    img_height, img_width = cimg.shape[0:2]
    
    h = math.ceil(img_height/9)
    w = math.ceil(img_width/9)
    for r in range(0,img_height, h):
        for c in range(0,img_width, w):
            cells.append(cimg[r:r+h , c:c+w , :])


############## Test on one image##############
load_imgdata()
img = imgs[1]
plt.imshow(img)

cimg = crop_sudoku(img)
plt.imshow(cimg)
############## Test End ######################




for cell in cells:
    plt.imshow(cell)
        

    



  # i) detect edges with "Canny algorithm"
    cimg_gray = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
    plt.imshow(cimg_gray) 

    cimg_edges = cv2.Canny(cimg_gray, 50, 200)
    plt.imshow(cimg_edges) 
    
    
    # ii) Detect points that form a line with "HoughLines" algorithm
    height, width = cimg_edges.shape
    minlength = 0.9*min(height,width)
    
    lines = cv2.HoughLinesP(cimg_edges, 1, np.pi/180, 100, minLineLength = minlength, maxLineGap=150)
    lines.shape

    lines[0][0][0]

    mask = np.zeros(cimg.shape, dtype=np.uint8) # create empty mask
    
    # Draw lines on grid and create matrix of lines
    lines_matrix = np.zeros((lines.shape[0], lines.shape[2]))
    lines_matrix.shape
    i = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(mask, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        lines_matrix[i,] = x1, y1, x2, y2 # create point matrices
        i += 1
    plt.imshow(mask)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   # sort
    _lines_x = []
    _lines_y = []
    for line in lines_matrix:
        orientation_i = math.atan2((line_i[0][1]-line_i[1][1]),(line_i[0][0]-line_i[1][0]))
        if (abs(math.degrees(orientation_i)) > 45) and abs(math.degrees(orientation_i)) < (90+45):
            _lines_y.append(line_i)
        else:
            _lines_x.append(line_i)

    _lines_x = sorted(_lines_x, key=lambda _line: _line[0][0])
    _lines_y = sorted(_lines_y, key=lambda _line: _line[0][1])

    merged_lines_x = merge_lines_pipeline_2(_lines_x)
    merged_lines_y = merge_lines_pipeline_2(_lines_y)


   

    # iv) calculate line from points
    x1, y1, x2, y2 = lines[0][0]
    
    
    
    
    # iv) get grid edges
    contours = get_contours(mask)
    len(contours)
    
    grid = cv2.drawContours(mask, contours,0,255,-1)
    plt.imshow(grid)
 
    
 
    
 
    
    # v) Sort by top to bottom and each row by left to right 
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(grid)
    vertical = np.copy(grid)
    
    ### horizonal lines ###
    cols = horizontal.shape[1]    # extract nr of cols 
    horizontal_size = cols // 10  # define desired nr of cols
       
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    
    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    horizontal = cv2.bitwise_not(horizontal)
    plt.imshow(horizontal)
    
    ### vertical lines ###
    # Specify size on vertical axis
    rows = vertical.shape[0]
    vertical_size = rows // 10
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical = cv2.bitwise_not(vertical)
    plt.imshow(vertical)


    


    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if h/w > 25: # vertical line
            cv2.drawContours(close,[cnt],0,255,-1)
        else:
            cv2.drawContours(close,[cnt],0,0,-1)
        
    
    
    
    
    
    
    
    
    
    
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
            cv2.drawCntours(mask, [c], -1, (255,255,255), -1)
            result = cv2.bitwise_and(img, mask)
            result[mask==0] = 255
            cv2.imshow('result', result)
            cv2.waitKey(175)
    
    plt.imshow('thresh', thresh)
    plt.imshow('invert', invert)






# might be useful somehow:
"""

 
"""


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
