import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import random
import numpy as np


# Create fucntions
def resizeFunc(x):
    dim = 50
    return cv2.resize(x, (dim, dim) , interpolation = cv2.INTER_CUBIC)

def changebackgroundFunc(x):
    max_grey = 85
    bgColor = int(round(random.uniform(0, max_grey), 0))
    mask = cv2.inRange(x, 0, bgColor)
    x[np.where(mask == 255)] = bgColor
    return x

def repositionandborderFunc(x):
    num_rows, num_cols = x.shape[:2]
    min_dark = 180
    bgColor = int(round(random.uniform(min_dark, 255), 0))
    column_shift = int(round(np.random.normal(0, 4), 0))
    row_shift = int(round(np.random.normal(0, 4), 0))
    translation_matrix = np.float32([[1, 0, column_shift], [0, 1, row_shift]])
    x = cv2.warpAffine(x, translation_matrix, (num_cols, num_rows))
    x[np.where(x == 0)] = bgColor
    return x

def addnoiseFunc(x):
    uniform_noise = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint8)
    cv2.randu(uniform_noise, 0, 255)
    ret, salt_noise = cv2.threshold(uniform_noise, 253, 255, cv2.THRESH_BINARY)
    salt_noise = (salt_noise).astype(np.uint8)
    x = cv2.subtract(x, salt_noise)
    uniform_noise = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint8)
    cv2.randu(uniform_noise, 0, 255)
    ret, pepper_noise = cv2.threshold(uniform_noise, 251, 255, cv2.THRESH_BINARY)
    pepper_noise = (pepper_noise).astype(np.uint8)
    x = cv2.add(x, pepper_noise)
    return x

def applyblurFunc(x):
   return cv2.GaussianBlur(x, (3, 3), 1)

def createAdditionalSamples(x, showImages):
    x = np.array(list(map(resizeFunc, x)))
    if showImages:
        plt.imshow(x, cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    x = np.array(list(map(changebackgroundFunc, x)))
    if showImages:
        plt.imshow(x, cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    x = np.array(list(map(repositionandborderFunc, x)))
    if showImages:
        plt.imshow(x, cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    x = np.array(list(map(addnoiseFunc, x)))
    if showImages:
        plt.imshow(x, cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    x = np.array(list(map(applyblurFunc, x)))
    if showImages:
        plt.imshow(x, cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    return x

#Modify MNIST Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

number_of_extends = 5
x_train_output = np.array(list(map(resizeFunc, x_train)))
y_train_output = y_train
x_test_output = np.array(list(map(resizeFunc, x_test)))
y_test_output = y_test

for i in range(1,number_of_extends):
    x_train_output = np.concatenate((x_train_output, createAdditionalSamples(x_train, False)), axis=0)
    y_train_output = np.concatenate((y_train_output, y_train), axis=0)
    x_test_output = np.concatenate((x_test_output, createAdditionalSamples(x_test, False)), axis=0)
    y_test_output = np.concatenate((y_test_output, y_test), axis=0)
    print(x_train_output.shape)

path = "D:\Programming\Python\SudokuSolver\data\moddedMNIST"
np.save(path + "x_train.npy", x_train_output)
np.save(path + "y_train.npy", y_train_output)
np.save(path + "x_test.npy", x_test_output)
np.save(path + "y_test.npy", y_test_output)