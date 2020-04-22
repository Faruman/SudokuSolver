################################################################
#            Identifying Sudoku Field                          #
################################################################

import os
os.getcwd()
os.chdir("C:\\Users\\sdien\\Documents\\GitHub\\SudokuSolver")

# import libraries
import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt


def load_imgdata():
    imgs = []
    path = "data\\sudoku_img\\mixed\\"
    for file in os.listdir(path):
        imgs.append(cv2.imread(os.path.join(path,file)))
    return imgs

# matplotlib summary
def summary(image):
    plt.imshow(image)
    print('Data Type: ', image.dtype)
    print('Dimensions: ', image.shape)
    

############## Image Processing ##############

# i) load data, convert to grayscale
imgs = load_imgdata()
summary(imgs[1]) 

img = imgs[1] 

# ii)  apply gray scale + further transformations, blur, dilate
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
"""img_blur = cv2.GaussianBlur(img, (3, 3), 0)

kernel = np.ones((5,5), np.uint8) 
img_dil = cv2.erode(img, None)

summary(img)
summary(img_blur)
summary(img_dil)"""


# iii) find contours with canny
img_edges = cv2.Canny(img,150,200)
img_edges = 
_,contours,_= cv2.findContours(img_edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
best_cnt = None
i = 0
for cnt in contours:
    i += 1
    print(i)
    x, y, w, h = cv2.boundingRect(cnt) # get contour coordinates
    area = cv2.contourArea(cnt)
    path = "data\\sudoku_img\\boxes\\"
    #cv2.drawContours(img, max_area_contour, 5, (255,0,0), 1)
    filename = "image"+str(i)+".jpg"
    cv2.imwrite(path+filename, img[y:y+h,x:x+w])
    if area > 1000:
        if area > max_area:
            max_area = area
            best_cnt = cnt
            print("New max area:", max_area, "Index:", i)


cv2.drawContours(mask,contours[330],0,255,4)
summary(mask)



# iv) Find lagest box from contours
_, contours, _ = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_area = []

    
# find contours with biggest area
max_area_index = contours_area.index(max(contours_area))
max_area_contour = contours[max_area_index]

# draw cnonour with biggest area
x, y, w, h = cv2.boundingRect(max_area_contour)

path = "data\\sudoku_img\\boxes\\image1.jpg"
cv2.drawContours(img, max_area_contour, 5, (255,0,0), 1)
#filename = "image"+str(i)+".jpg"
cv2.imwrite(path, img[y:y+h,x:x+w])

summary(img)


largest_area=0
# approx rectangle directly
for cnt in contours:
     
    if (area > largest_area):
        largest_area=area
        bounding_rect=cv2.boundingRect(contours[cnt])

summary(img)
    









cropped_dir_path = "\\data\\sudoku_img\boxes"
idx = 0
for c in contours:
    # Returns the location and width,height for every contour
    x, y, w, h = cv2.boundingRect(c)
    if (w > 80 and h > 20) and w > 3*h:
        idx += 1
        new_img = img[y:y+h, x:x+w]
        if not cv2.imwrite(os.path.join(cropped_dir_path,'.jpg'), new_img):
            raise Exception("Could not write image")
# If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
    if (w > 80 and h > 20) and w > 3*h:
        idx += 1
        new_img = img[y:y+h, x:x+w]
        if not cv2.imwrite(cropped_dir_path+str(idx) + '.jpg', new_img):
            raise Exception("Could not write image")

    	

# Crop sudoku
height, width = img.shape[:2] # get height and width of image

start_row, start_col=int(height*.25),int(width*.25) # pick corners
end_row, end_col=int(height*.75),int(width*.75)

cropped=img[start_row:end_row, start_col:end_col] # crop image
summary(img)
summary(cropped)





# find contours in the edged image, keep only the largest ones, and initialize our screen contour
cnts = cv2.findContours(img_edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()


cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None

cv2.drawContours(img,cnts,-1,(255,0,0),3)
summary(img)




# v) find correct contour


# save variable
img_post = img
img = img_post











# Hough Algorithm
for line in lines[0:1]:       
    # get coordinates of lines
    rho, theta = line[0]
    print("rho ", rho, "theta ",theta)
    a = np.cos(theta)
    b = np.sin(theta)
    print("a: ", a, "b: ", b)
    x0 = a * rho
    y0 = b * rho
    print("x0 ", x0, "y0 ",y0)
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    print("x1 ", x1, "y1 ",y1)
    x2 = int(x0 + 1000 * (-b))
    y2 = int(y0 + 1000 * (a))
    print("x2 ", x2, "y2",y1)
    # plot lines on image  (image, point1, point2, color, line thickness) 
    line = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 5) 
    

cv2.imwrite('houghlines3.jpg',img)

img = cv2.line(img, (400,400), (250, 250), (255,0,0), 5)
summary(img)


# compare Canny vs original



plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()




# see later if that works better:
img = cv2.GaussianBlur(img.copy(), (5, 5), 0) # apply blur
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) 
    # specify adaptive gaussian threshhold: see https://bit.ly/2xN2v3I)
    
img = cv2.bitwise_not(img, img)  # turn image into binary
kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8) 
img = cv2.dilate(img, kernel) # dilate to reduce noise
    
   
plt.imshow(img, cmap = 'gray')
       







##############################################################
# iii) apply threshold to image (OR TRY USING CANNY EDGE DETECTION HERE)
(thresh, img_bin) = cv2.threshold(img, 125, 255,cv2.THRESH_BINARY)

# convert to grayscale and invert image
img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
img_bin = 255-img_bin 
cv2.imwrite("Image_bin.jpg",img_bin)

summary(img_bin)

# iv) define kernels
kernel_length = np.array(img).shape[1]//80 # Defining a kernel length

# verticle kernel of (1 X kernel_length) - Help detect verticle lines
verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
# horizontal kernel of (kernel_length X 1) - Help detect horizontal lines
hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

# A kernel of (3 X 3) ones.
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# v) Detect vertical and horizontal lines
# Morphological operation to detect vertical lines from an image
img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
cv2.imwrite("verticle_lines.jpg",verticle_lines_img)
summary(verticle_lines_img)

# Morphological operation to detect horizontal lines from an image
img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)
summary(horizontal_lines_img)

# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
alpha = 0.5
beta = 1.0 - alpha
# This function helps to add two image with specific weight parameter to get a third image as summation of two image.
img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
(thresh, img_final_bin) = cv2.threshold(img_final_bin, 160,205, cv2.THRESH_BINARY)
cv2.imwrite("img_final_bin.jpg",img_final_bin)
summary(img_final_bin)

    
def sort_contours(cnts, method="left-to-right"):
	i = 0
	if method == "top-to-bottom":
		i = 1
	# construct the list of bounding boxes and sort them from top to bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i]))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
   