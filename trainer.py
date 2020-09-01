"""
Training script for the model.
"""

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from libs.dloader import DataLoader
from libs.conv_model import ConvModel, createConvModel2
from libs.callbacks import load_cb

DATASET_PATH = './dataset/'
NUM_EPOCHS = 40
BATCH_SIZE = 32
NUM_CLASSES = 5


def plot_confusion_matrix(confusion_mat: np.array) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    df_cm = pd.DataFrame(confusion_mat.astype(np.int), range(NUM_CLASSES), range(NUM_CLASSES))
    ax = sn.heatmap(df_cm, annot=True, fmt='d')
    ax.set_ylim(len(confusion_mat), 0.0)


def train_model(model: tf.keras.Model, loader: DataLoader):
    train_set, test_set = loader.open()
    train_set = train_set.cache().batch(BATCH_SIZE).repeat()
    test_set = test_set.cache().batch(BATCH_SIZE)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=10,
                                          restore_best_weights=True,
                                          verbose=True)
    training_history = model.fit(train_set, epochs=NUM_EPOCHS, validation_data=test_set,
                                 steps_per_epoch=10, class_weight=class_weights,
                                 callbacks=[es])
    return training_history


def generate_multiobjective_labels(images: [tf.Tensor], labels: [int]):
    ...


if __name__ == '__main__':
    # Setup GPU usage for TensorFlow, if available
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # tf.debugging.set_log_device_placement(True)
    tf.keras.backend.set_floatx('float64')
    # Setup random seed
    tf.random.set_seed(42)
    np.random.seed(42)

    loader = DataLoader(DATASET_PATH, train_split=0.7, augment=True, callbacks={'load': load_cb})
    train_set, test_set = loader.open()
    train_set = train_set.cache().batch(BATCH_SIZE).repeat()
    test_set = test_set.batch(BATCH_SIZE)

    #model = ConvModel()
    model = createConvModel2()
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.compile(optimizer='adam', loss='kullback_leibler_divergence', metrics=['acc'])

    class_weights = {0: 0.3,
                     1: 5.,
                     2: 4.,
                     3: 2.,
                     4: 2.}

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                          patience=7,
                                          restore_best_weights=True,
                                          verbose=True)
    training_history = model.fit(train_set, epochs=NUM_EPOCHS, validation_data=test_set,
                                 steps_per_epoch=10, #class_weight=class_weights,
                                 callbacks=[es])

    # Fill confusion matrix
    cm = np.zeros(shape=(NUM_CLASSES, NUM_CLASSES))
    for x_val, y_val in test_set.take(10):
        y_true = np.argmax(y_val, axis=1)
        y_pred = np.argmax(model.predict(x_val), axis=1)
        tcm = confusion_matrix(y_true, y_pred, labels=range(0, NUM_CLASSES))
        cm += tcm

    plot_confusion_matrix(cm)
    plt.tight_layout()
    plt.show()

    exit(0)
