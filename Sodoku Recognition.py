
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:01:32 2020

@author: sdien
"""

################################################################
#            Identifying Sudoku Field                          #
################################################################



# import libraries

import os
os.getcwd()
os.chdir("C:\\Users\\sdien\\Documents\\GitHub\\SudokuSolver")

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import image
from matplotlib import pyplot


import cv2
import PIL
from PIL import Image

print('Pillow Version:', PIL.__version__)


# image summary 
def summary(image):
    stats = {"Image Format": image.format,
             "Mode": image.mode,
             "Dimensions": image.size}
    for i in stats:
        print(i, ":", stats[i])
       # image.show()



# load the image
filepath = "data/sudoku_img/v2_train/image1.jpg"
image = Image.open(filepath)



    # show the image
summary(image)




