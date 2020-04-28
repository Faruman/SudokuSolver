
##############################################################################
#                    Model for Recognizing Digits                            #
##############################################################################

#%% Setup 

import os
os.getcwd()
os.chdir("C:\\Users\\sdien\\Documents\\GitHub\\SudokuSolver")

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, losses

#%% Define functions
def load_MNIST():
    global x_train, y_train, x_test, y_test 
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

    return x_train, y_train, x_test, y_test
        
def create_model():
    tf.keras.backend.clear_session() # closes existing sessions
    
    # Build model
    model = models.Sequential([    
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),  
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu' ),
        layers.Dense(10, activation='softmax' )
        ])
        
    # Compile model
    model.compile(optimizer='adam',
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

    
    
#%% Build model

   
load_MNIST()
plt.imshow(x_train[1].reshape((28,28)))

# Build Model
ConvModel = create_model()
ConvModel.summary()

tf.strings.reduce_joinas_str(ConvModel)
# Train Model
ConvModel.fit(x_train, y_train, batch_size= 32, epochs = 3, validation_split = 0.1)

# Evaluate Model
preds =  ConvModel.evaluate(x = x_test, y = y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

# save model
ConvModel.reset_metrics() 
ConvModel.save('model', save_format='tf') #  Save the model













