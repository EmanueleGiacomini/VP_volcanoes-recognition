"""
conv_model module
Contains the definition of the CNN (Convolutional Neural Network) model for Venus Volcanoes
Dataset.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization


class ConvModel(tf.keras.Model):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = Conv2D(32, (5, 5), activation='relu', input_shape=(128, 128), kernel_regularizer='l2')
        self.drop1 = Dropout(rate=0.05)
        self.mpool1 = MaxPooling2D()
        self.conv2 = Conv2D(64, (3, 3), activation='relu', kernel_regularizer='l2')
        self.drop2 = Dropout(rate=0.05)
        self.mpool2 = MaxPooling2D()
        self.conv3 = Conv2D(128, (3, 3), activation='relu', kernel_regularizer='l2')
        self.drop3 = Dropout(rate=0.05)
        self.mpool3 = MaxPooling2D()
        self.flat1 = Flatten()
        self.dense1 = Dense(32, activation='relu', kernel_regularizer='l2')
        self.drop4 = Dropout(rate=0.05)
        self.dense2 = Dense(5, activation='softmax', kernel_regularizer='l2')

    @tf.function
    def call(self, inputs, training):
        x = self.conv1(inputs)
        x = self.drop1(x, training)
        x = self.mpool1(x)
        x = self.conv2(x)
        x = self.drop2(x, training)
        x = self.mpool2(x)
        x = self.conv3(x)
        x = self.drop3(x, training)
        x = self.mpool3(x)
        x = self.flat1(x)
        x = self.dense1(x)
        x = self.drop3(x, training)
        return self.dense2(x)


def createConvModel2():
    model = tf.keras.models.Sequential()
    model.add(Conv2D(2, (3, 3)))
    model.add(Conv2D(3, (3, 3)))
    model.add(Conv2D(8, (5, 5)))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (5, 5)))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D())
    model.add(Conv2D(24, (7, 7)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(5, activation='softmax'))
    return model


class ConvModel3(tf.keras.Model):
    def __init__(self):
        super(ConvModel, self).__init__()
        ...

    def call(self, inputs):
        ...
