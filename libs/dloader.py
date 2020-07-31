"""
dloader module
The DataLoader class is used to load in memory, previously stored datasets generated by a DataBuilder object
"""

import numpy as np
import tensorflow as tf
from PIL import Image

LABELS_PATH = 'labels.csv'

def normalize_image(image: tf.Tensor) -> tf.Tensor:
    return tf.image.per_image_standardization(image)

DT_SHUFFLE_BUF_SIZE = int(1e4)
NUM_CLASSES = 5

class DataLoader:
    def __init__(self, path: str, train_split=0.7, callbacks={}):
        # Open metadata (labels) file
        with open(path + LABELS_PATH, 'r') as f:
            data = []
            for line in f.readlines():
                data.append(int(line.strip()))
        labels_tf = tf.convert_to_tensor(np.array(data), tf.int16)
        images_tf_raw = []
        # Read images
        for i in range(labels_tf.shape[0]):
            # generate image path
            impath = path + str(i) + '.jpg'
            image = np.array(Image.open(impath))
            image = tf.convert_to_tensor(image, tf.float64)
            image = tf.expand_dims(image, -1)
            images_tf_raw.append(image)
            if 'load' in callbacks:
                callbacks['load'](i, labels_tf.shape[0]-1, 'Loading images')
        # Normalize images
        images_tf = []
        for i, image in enumerate(images_tf_raw):
            images_tf.append(normalize_image(image))
            if 'load' in callbacks:
                callbacks['load'](i, len(images_tf_raw)-1, 'Normalizing images')

        self.images = images_tf
        # OneHotEncode labels
        self.labels = tf.keras.utils.to_categorical(labels_tf, NUM_CLASSES)
        data_tf_cmpl = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        data_tf_cmpl = data_tf_cmpl.shuffle(DT_SHUFFLE_BUF_SIZE)
        # Split train/test sets
        train_size = int(len(self.images) * train_split)
        self.train_size = train_size
        self.test_size = len(self.images) - self.train_size
        train_dataset = data_tf_cmpl.take(train_size)
        test_dataset = data_tf_cmpl.skip(train_size)
        self.train, self.test = train_dataset, test_dataset

    def open(self):
        return self.train, self.test

from matplotlib import pyplot as plt
from libs.callbacks import load_cb

if __name__ == '__main__':
    loader = DataLoader('./dataset/', callbacks={'load': load_cb})
    train_set, test_set = loader.open()

    for x, y in train_set.take(1):
        print(x)
        print(f'y : {y}')