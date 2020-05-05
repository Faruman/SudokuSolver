import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt

class Complex2_Net(nn.Module):
    def __init__(self):
        super(Complex2_Net, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(128 * 13 * 13, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        return x

def extract_sudoku(img):
    # Detect edges with "Canny Edge Detector"
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray
    img_edges = cv2.Canny(img_gray, 30, 150)  # detects edges in image
    # Find largest contour
    contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # get contours
    largest_cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # sort by area and take first element

    # Draw largest rectangle
    rect = np.zeros(img.shape[0:2], dtype=np.uint8)
    rect = cv2.drawContours(rect, [largest_cnt], 0, (255), 2)

    # Find corners points with "Shi-Tomasi Corner Detector"
    pts = cv2.goodFeaturesToTrack(rect, minDistance=20, maxCorners=4, qualityLevel=0.01)
    pts = np.int0(pts)
    pts = pts.reshape((4, 2))

    # Sort corner points
    s = pts.sum(axis=1)  # x + y
    tl = tuple(pts[np.argmin(s)])  # lower sum; top left
    br = tuple(pts[np.argmax(s)])  # higer sum; bottom right

    diff = np.diff(pts, axis=1)  # x - y
    tr = tuple(pts[np.argmin(diff)])  # lower diff; top right
    bl = tuple(pts[np.argmax(diff)])  # higer diff; bottom left

    # Adjust perspective
    rows = cols = 500

    pts1 = np.float32([tl, tr, bl, br])
    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    cimg = cv2.warpPerspective(img, M, (cols, rows))

    return cimg


def extract_numbers(cimg):
    #load my model
    model = Complex2_Net()
    model.load_state_dict(torch.load("D:/Programming/Python/SudokuSolver/data/moddedMNIST/model_complex_v2_ext.sav"))
    model.eval()

    # Split image into 81 cells with numbers
    nums = []
    img_height, img_width = cimg.shape[0:2]
    h = int(img_height / 9) + 1
    w = int(img_width / 9) + 1
    for r in range(0, img_height, h):
        for c in range(0, img_width, w):
            nums.append(cimg[r:r + h, c:c + w])

    # Clean numbers
    preds  = np.zeros((9,9))
    num_height = num_width = 50

    inputs = torch.FloatTensor()
    for num in nums:
        img = cv2.resize(num, (num_height, num_width), interpolation=cv2.INTER_CUBIC)  # fix size to 50, 50
        img = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype('float32')
        ten = torch.tensor(img) / 255
        ten = ten.unsqueeze(0).unsqueeze(0)
        #plt.imshow(img, cmap='gray_r', vmin=0, vmax=255)
        #plt.show()
        #print(model(ten))
        #print(model(ten).cpu().data.max(1, keepdim=True)[1])
        inputs = torch.cat((inputs, ten), dim=0)

    outputs = model(inputs).cpu().data.max(1, keepdim=True)
    preds = outputs[1].numpy().reshape((9, 9))
    probs = outputs[0].numpy().reshape((9, 9))

    return preds, probs


raw_img = cv2.imread("D:\Programming\Python\SudokuSolver\data\sudoku_img\mixed\image (3).jpg")
plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY).astype('float32'), cmap='gray_r', vmin=0, vmax=255)
plt.show()
cimg = extract_sudoku(raw_img)
plt.imshow(cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY).astype('float32'), cmap='gray_r', vmin=0, vmax=255)
plt.show()
sudoku = extract_numbers(cimg)
print(sudoku)