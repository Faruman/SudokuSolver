import tensorflow as tf
import cv2
from matplotlib import pyplot as plt
import random
import numpy as np


# Create fucntions
def resizeFunc(x):
    dim = 50
    return cv2.resize(x, (dim, dim) , interpolation = cv2.INTER_CUBIC)

def generateComputerImage(number):
    font = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX][random.randint(0,3)]
    fontsize = [2, 3][random.randint(0,1)]
    dim = 50
    image = np.zeros([dim, dim], dtype=np.uint8)
    image = cv2.putText(image, str(number), (int(round(0.15*dim, 0)),int(dim-round(0.15*dim, 0))), font, 1.75, 255, fontsize)
    return image

def shrinkFunc(x):
    num_rows, num_cols = x.shape[:2]
    bgColor = 0
    top = bottom = left = right = abs(np.random.normal(0.25, 0.25))
    x  = cv2.copyMakeBorder(x, int(round(top*num_rows, 0)), int(round(bottom*num_rows, 0)), int(round(left*num_cols, 0)), int(round(right*num_cols, 0)), cv2.BORDER_CONSTANT);
    return x

def changebackgroundFunc(x):
    max_grey = 125
    bgColor = int(round(random.uniform(25, max_grey), 0))
    mask = cv2.inRange(x, 0, bgColor)
    x[np.where(mask == 255)] = bgColor
    return x

def changenumberFunc(x):
    min_grey = 155
    bgColor = int(round(random.uniform(min_grey, 255), 0))
    mask = cv2.inRange(x, bgColor, 255)
    x[np.where(mask == 255)] = bgColor
    return x

def repositionandborderFunc(x):
    num_rows, num_cols = x.shape[:2]
    min_dark = 155
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
    ret, salt_noise = cv2.threshold(uniform_noise, 254, 255, cv2.THRESH_BINARY)
    salt_noise = (salt_noise).astype(np.uint8)
    x = cv2.subtract(x, salt_noise)
    uniform_noise = np.zeros((x.shape[0], x.shape[1]), dtype=np.uint8)
    cv2.randu(uniform_noise, 0, 255)
    ret, pepper_noise = cv2.threshold(uniform_noise, 253, 255, cv2.THRESH_BINARY)
    pepper_noise = (pepper_noise).astype(np.uint8)
    x = cv2.add(x, pepper_noise)
    return x

def applyblurFunc(x):
   return cv2.GaussianBlur(x, (3, 3), 1)

def invertFunc(x):
    return cv2.bitwise_not(x)

def createAdditionalSamples(x, showImages):
    i = int(round(random.uniform(0, x.shape[0]), 0))
    x = np.array(list(map(shrinkFunc, x)))
    if showImages:
        plt.imshow(x[i], cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    x = np.array(list(map(resizeFunc, x)))
    if showImages:
        plt.imshow(x[i], cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    x = np.array(list(map(changebackgroundFunc, x)))
    if showImages:
        plt.imshow(x[i], cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    x = np.array(list(map(changenumberFunc, x)))
    if showImages:
        plt.imshow(x[i], cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    x = np.array(list(map(repositionandborderFunc, x)))
    if showImages:
        plt.imshow(x[i], cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    x = np.array(list(map(addnoiseFunc, x)))
    if showImages:
        plt.imshow(x[i], cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    x = np.array(list(map(applyblurFunc, x)))
    if showImages:
        plt.imshow(x[i], cmap='gray_r', vmin=0, vmax=255)
        plt.show()
    return x

#Modify MNIST Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#resize images
x_train = np.array(list(map(resizeFunc, x_train)))
x_test = np.array(list(map(resizeFunc, x_test)))

#add artificial samples to have computer written images
temp_y_train = np.random.randint(low=0, high=9, size=int(round(0.1*x_train.shape[0], 0)))
x_train = np.concatenate((x_train, np.array(list(map(generateComputerImage, temp_y_train)))), axis=0)
y_train = np.concatenate((y_train, temp_y_train), axis=0)
temp_y_test = np.random.randint(low=0, high=9, size=int(round(0.1*x_test.shape[0], 0)))
x_test = np.concatenate((x_test, np.array(list(map(generateComputerImage, temp_y_test)))), axis=0)
y_test = np.concatenate((y_test, temp_y_test), axis=0)

#replace zeros from train and test set with empty images
x_train[np.where(y_train == 0)] = np.zeros([len(np.where(y_train == 0)[0]), 50, 50],dtype=np.uint8)
x_test[np.where(y_test == 0)] = np.zeros([len(np.where(y_test == 0)[0]), 50, 50],dtype=np.uint8)

#create new samples
number_of_extends = 10
inverted_training = True
inverted_test = False
x_train_output = x_train
y_train_output = y_train
x_test_output = x_test
y_test_output = y_test

for i in range(1,number_of_extends):
    x_train_output = np.concatenate((x_train_output, createAdditionalSamples(x_train, True)), axis=0)
    y_train_output = np.concatenate((y_train_output, y_train), axis=0)
    x_test_output = np.concatenate((x_test_output, createAdditionalSamples(x_test, True)), axis=0)
    y_test_output = np.concatenate((y_test_output, y_test), axis=0)
    print(x_train_output.shape)

if inverted_training == True:
    x_train_output = np.concatenate((x_train_output, np.array(list(map(invertFunc, x_train_output)))), axis=0)
    y_train_output = np.concatenate((y_train_output, y_train_output), axis=0)
if inverted_test == True:
    x_test_output = np.concatenate((x_test_output, np.array(list(map(invertFunc, x_test_output)))), axis=0)
    y_test_output = np.concatenate((y_test_output, y_test_output), axis=0)

path = "D:/Programming/Python/SudokuSolver/data/moddedMNIST/"
np.save(path + "x_train_ext3.npy", x_train_output)
np.save(path + "y_train_ext3.npy", y_train_output)
np.save(path + "x_test_ext3.npy", x_test_output)
np.save(path + "y_test_ext3.npy", y_test_output)