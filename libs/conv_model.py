"""
conv_model module
Contains the definition of the CNN (Convolutional Neural Network) model for Venus Volcanoes
Dataset.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten

class ConvModel(tf.keras.Model):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64))
        self.mpool1 = MaxPooling2D()
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.flat1 = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(5, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.mpool1(x)
        x = self.conv2(x)
        x = self.flat1(x)
        x = self.dense1(x)
        return self.dense2(x)
