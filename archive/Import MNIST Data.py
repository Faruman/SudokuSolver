# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 11:42:30 2020

@author: sdien
"""
# Setup
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


import os
os.getcwd()
os.chdir("C:\\Users\\sdien\\Documents\\GitHub\\SudokuSolver")


# Import function MNIST digits
def load_mnist_dataset():
    path = os.getcwd() + "\\data\\MNIST\\" # path where data is stored
    def download(filename, source = 'http://yann.lecun.com/exdb/mnist/'):
        print("Downloading ", filename)
        import urllib
        urllib.request.urlretrieve(source+filename, path+filename)
    # downloads digit images from Yann Lecun website and stores it on local drive

    import gzip
    
    def load_mnist_images(filename):
        if not os.path.exists(path+filename):
            download(filename)
        # checks if file is on drive, if not the file ii downlaoded
        with gzip.open(path+filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset = 16)
            # Opens zip file and extract 1D np array from zip file 
            
            data = data.reshape(-1, 1, 28, 28)
            # reshape 1D np array into image format
            # Dim1: Nr of images: -1 because infered from other vars
            # Dim2: Nr of cahnnels: 1 because only grey scale
            # Dim3-4: Nr of pixels: 28x28
        return data / np.float32(256)
        # converts the byte value to a float32
    def load_mnist_labels(filename):
        if not os.path.exists(path+filename):
            download(filename)
        # checks if file is on drive, if not the file ii downlaoded
        with gzip.open(path+filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset = 8)
                # Opens zip file and extract labels as 1D np array from zip file      
        return data
    
    # Load images & labels for train & test
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    Y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    Y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = load_mnist_dataset()


# Display image function
def img_show(img):
    img = img.reshape([28,28])
    print(plt.imshow(img, cmap='gray_r'))
    
img_show(X_train[0])




