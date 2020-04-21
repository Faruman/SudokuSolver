# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 13:01:32 2020

@author: sdien
"""

################################################################
#            ConvNN for Recognizing Digits                     #
################################################################




import os
os.getcwd()
os.chdir("C:\\Users\\sdien\\Documents\\GitHub\\SudokuSolver")

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, losses

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0
# add 4th dimension to match Conv layer (expects another dim for RGB channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Vizualize 10 examples
ROW = 4
COLUMN = 5
for i in range(ROW * COLUMN):
    image = x_train[i].reshape((28,28))
    plt.subplot(ROW, COLUMN, i+1)        # subplot with size
    plt.imshow(image, cmap='gray_r')     # cmap='gray_r' is for black and white picture.
    plt.title('label = {}'.format(y_train[i]))
    plt.axis('off')  # do not show axis value

# Clear old Session
tf.keras.backend.clear_session()

# Build Model
model = models.Sequential()

model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))     
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu' ))
model.add(layers.Dense(10, activation='softmax' ))

model.summary()

# Compile Model
model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train Model
model.fit(x_train, y_train, batch_size= 32, epochs = 10, validation_split = 0.1)

