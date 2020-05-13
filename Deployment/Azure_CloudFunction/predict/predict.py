import os
import numpy as np
import base64
import cv2
import torch
from .models import Sudoku_Net

class sudokuPrediction():
    def __init__(self, model_path):
        model = Sudoku_Net()
        state_dict = torch.load(os.path.join(model_path), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.to('cpu')
        model.eval()
        self._model = model

    def _postprocess(self, outputs):
        outputs = outputs.data.max(1, keepdim=True)

        pred = outputs[1].numpy().reshape((9, 9))
        prob = outputs[0].numpy().reshape((9, 9))

        puzzle = list()
        _, pred_counts = np.unique(pred[pred != 0], return_counts=True)
        if np.mean(prob) > 0.5 and np.count_nonzero(pred) > 16 and max(pred_counts) <= 9:
            puzzle = pred.tolist()

        return (puzzle)

    @staticmethod
    def _preprocess(raw_img):
        decoded_img = base64.b64decode(raw_img)
        np_img = np.fromstring(decoded_img, dtype= np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)

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

        return cimg

    def predict(self, encoded, **kwargs):
        # Pass in a base64 string encoding of the image
        img = self._preprocess(encoded)

        # Split image into 81 cells with numbers
        nums = []
        img_height, img_width = img.shape[0:2]
        h = int(img_height / 9) + 1
        w = int(img_width / 9) + 1
        for r in range(0, img_height, h):
            for c in range(0, img_width, w):
                nums.append(img[r:r + h, c:c + w])

        # Clean numbers
        num_height = num_width = 50

        inputs = torch.FloatTensor()
        for num in nums:
            img = cv2.resize(num, (num_height, num_width), interpolation=cv2.INTER_CUBIC)  # fix size to 50, 50
            img = cv2.bitwise_not(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype('float32')
            ten = torch.tensor(img) / 255
            ten = ten.unsqueeze(0).unsqueeze(0)
            inputs = torch.cat((inputs, ten), dim=0)

        outputs = self._model.forward(inputs).cpu()
        puzzle = self._postprocess(outputs)
        return {"puzzle": puzzle}