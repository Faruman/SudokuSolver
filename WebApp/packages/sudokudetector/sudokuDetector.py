import numpy as np
import cv2
import torch
import torch.nn as nn

from matplotlib import pyplot as plt

class Complex_Net(nn.Module):
    def __init__(self):
        super(Complex_Net, self).__init__()

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

class sudokuProcessor():
    def __init__(self, model_path):
        self.img_raw = np.array([])
        self.img_processed = np.zeros((500, 500))
        self.pred = np.zeros((9, 9))
        self.pred = np.zeros((9, 9))
        self.model = Complex_Net()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def extract_sudoku(self, img = np.array([])):
        if img.size == 0:
            img = self.img_raw
        else:
            self.img_raw = img
        # Detect edges with "Canny Edge Detector"
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
        img_edges = cv2.Canny(img_blur, 30, 200)  # detects edges in image

        kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # define cross kernel
        img_morph = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, kernel_cross)  # close grid lines

        # Find largest contour
        contours, _ = cv2.findContours(img_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # get contours
        cnt = max(contours, key=cv2.contourArea)  # find largest contour by area
        cnt = cnt.reshape((-1, 2))

        # Find corner points of contour
        s = np.sum(cnt, axis=1)  # x + y
        tl = tuple(cnt[np.argmin(s)])  # lower sum; top left
        br = tuple(cnt[np.argmax(s)])  # higher sum; bottom right

        diff = np.diff(cnt, axis=1)  # x - y
        tr = tuple(cnt[np.argmin(diff)])  # lower diff; top right
        bl = tuple(cnt[np.argmax(diff)])  # higher diff; bottom left

        # Adjust perspective
        rows = cols = 500

        pts1 = np.float32([tl, tr, bl, br])
        pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        cimg = cv2.warpPerspective(img, M, (cols, rows))

        self.img_processed = cimg


    def extract_numbers(self, cimg = np.array([])):
        if cimg.size == 0:
            cimg = self.img_processed
        else:
            self.img_processed = cimg

        #load my model
        self.model.eval()

        # Split image into 81 cells with numbers
        nums = []
        img_height, img_width = cimg.shape[0:2]
        h = int(img_height / 9) + 1
        w = int(img_width / 9) + 1
        for r in range(0, img_height, h):
            for c in range(0, img_width, w):
                nums.append(cimg[r:r + h, c:c + w])

        # Clean numbers
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

        outputs = self.model(inputs).cpu().data.max(1, keepdim=True)
        self.pred = outputs[1].numpy().reshape((9, 9))
        self.prob = outputs[0].numpy().reshape((9, 9))

    def process(self, img):
        self.img_raw = img
        self.extract_sudoku()
        self.extract_numbers()
        return(self.pred, self.prob)
