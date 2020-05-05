##############################################################################
#                      Identifying Sudoku Field                              #
##############################################################################

# %% Setup
import os
import numpy as np
import cv2

os.getcwd()
os.chdir("C:\\Users\\sdien\\Documents\\GitHub\\SudokuSolver")


# %% Define functions

def show(img):
    # resize but keep aspect ratio
    h, w = img.shape[0:2]
    ratio = h / w

    new_h = 500
    new_w = int(new_h * ratio)
    img = cv2.resize(img, (new_w, new_h))

    # show img (hit enter to close window)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_imgdata():
    global imgs, imgs_upload
    imgs = []
    imgs_upload = []
    path1 = "data\\sudoku_img\\mixed\\"
    path2 = "data\\sudoku_img\\uploads\\"
    for file in os.listdir(path1):
        imgs.append(cv2.imread(os.path.join(path1, file)))
    for file in os.listdir(path2):
        imgs_upload.append(cv2.imread(os.path.join(path2, file)))
    return imgs, imgs_upload


def extract_sudoku(img):
    # Detect edges with "Canny Edge Detector"
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_edges = cv2.Canny(img_blur, 30, 200)  # detects edges in image

    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # define cross kernel
    img_morph = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, kernel_cross)  # close grid lines

    # Find largest contour
    _, contours, _ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # get contours
    cnt = max(contours, key=cv2.contourArea)  # find largest contour by area
    cnt = cnt.reshape((-1, 2))

    # Find corner points of contour
    s = np.sum(cnt, axis=1)  # x + y
    tl = tuple(cnt[np.argmin(s)])  # lower sum; top left
    br = tuple(cnt[np.argmax(s)])  # higer sum; bottom right

    diff = np.diff(cnt, axis=1)  # x - y
    tr = tuple(cnt[np.argmin(diff)])  # lower diff; top right
    bl = tuple(cnt[np.argmax(diff)])  # higer diff; bottom left

    """# Draw largest rectangle with corner poitns
    rect = np.zeros(img.shape[0:2], dtype=np.uint8)
    rect = cv2.drawContours(rect, [cnt], 0, (255), 2)
    for (x,y) in [tl,tr,bl,br]:
        rect = cv2.circle(rect, (x,y), 10, 255, thickness = -1)
    show(rect)
    """

    # Adjust perspective 
    rows = cols = 500

    pts1 = np.float32([tl, tr, bl, br])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    cimg = cv2.warpPerspective(img, M, (cols, rows))

    return cimg


def extract_numbers(cimg):
    # Split image into 81 cells with numbers 
    nums = []
    img_height, img_width = cimg.shape[0:2]
    h = int(img_height / 9) + 1
    w = int(img_width / 9) + 1
    for r in range(0, img_height, h):
        for c in range(0, img_width, w):
            nums.append(cimg[r:r + h, c:c + w])

    # Clean numbers 
    nums_processed = []
    num_height = num_width = 100
    for num in nums:
        n_res = cv2.resize(num, (num_height, num_width), interpolation=cv2.INTER_CUBIC)  # fix size to 50,50
        n_blur = cv2.medianBlur(n_res, 3)
        n_gray = cv2.cvtColor(n_blur, cv2.COLOR_BGR2GRAY)  # convert to gray

        nums_processed.append(n_gray)

    return nums_processed


def list81_to_image(list81):
    h, w = list81[0].shape[0:2]
    plot_height, plot_width = np.multiply(list81[0].shape[0:2], 9)

    plot = np.zeros((plot_height, plot_width), np.uint8)

    # overwrite plot with images from list    
    i = 0
    for r in range(0, plot_height, h):
        for c in range(0, plot_width, w):
            (plot[r:r + h, c:c + w]) = list81[i]
            i += 1

    return plot


# %% TEST

# Test ONE   
load_imgdata()

img = imgs[54]  # somewhat working example: imgs[1], img_uplaod[1,3]
cimg = extract_sudoku(img)
show(img)
show(cimg)

numbers_list = extract_numbers(cimg)
numbers_plot = list81_to_image(numbers_list)
show(numbers_plot)

# Test ALL
for img in imgs:
    cimg = extract_sudoku(img)

    show(cimg)

print("hi")
