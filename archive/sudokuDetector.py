import numpy as np
import cv2
from matplotlib import pyplot as plt

#functions
def shLargestFlood(matrix):
    count = 0
    max =-1
    for y in range(0, matrix.shape[0]):
        for x in range(0, matrix.shape[1]):
            if matrix[y][x] >= 128:
                area,_ ,_ ,_ = cv2.floodFill(matrix, None, (x,y), 64)
                if (area > max):
                    maxPt = (x, y)
                    max = area

    return maxPt, matrix

def shGetRidOfColor(matrix, max, color):
    for y in range(0, matrix.shape[0]):
        for x in range(0, matrix.shape[1]):
            if matrix[y][x] == color and y != max[0] and x != max[1]:
                cv2.floodFill(matrix, None, (x, y), 0)
    return matrix

sudoku = cv2.imread("D:/Programming/Python/SudokuSolver/data/sudoku_img/uploads/5.png", 0)
height = sudoku.shape[0]
width = sudoku.shape[1]
plt.imshow(sudoku, cmap='gray', vmin=0, vmax=255)
plt.show()

outerBox = cv2.GaussianBlur(sudoku, (11, 11), 0)
outerBox = cv2.adaptiveThreshold(outerBox, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
outerBox = cv2.bitwise_not(outerBox)
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8)
outerBox = cv2.dilate(outerBox, kernel,iterations = 1)
plt.imshow(outerBox, cmap='gray', vmin=0, vmax=255)
plt.show()

maxPt, outerBox = shLargestFlood(outerBox)
cv2.floodFill(outerBox, None, maxPt, 255)
plt.imshow(outerBox, cmap='gray', vmin=0, vmax=255)
plt.show()

outerBox = shGetRidOfColor(outerBox, maxPt, 64)
plt.imshow(outerBox, cmap='gray', vmin=0, vmax=255)
plt.show()

outerBox = cv2.erode(outerBox, kernel, iterations= 1)
plt.imshow(outerBox, cmap='gray', vmin=0, vmax=255)
plt.show()

cv2.HoughLines(outerBox, lines, 1, )

