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
from libs.conv_model import ConvModel
from libs.callbacks import load_cb

DATASET_PATH = './dataset/'
NUM_EPOCHS = 20
BATCH_SIZE = 32
NUM_CLASSES = 5


def plot_confusion_matrix(confusion_mat: np.array) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    df_cm = pd.DataFrame(confusion_mat.astype(np.int), range(NUM_CLASSES), range(NUM_CLASSES))
    ax = sn.heatmap(df_cm, annot=True, fmt='d')
    ax.set_ylim(len(confusion_mat), 0.0)



if __name__ == '__main__':
    # Uncomment only for GPU-usage of TensorFlow

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    #tf.debugging.set_log_device_placement(True)

    tf.keras.backend.set_floatx('float64')


    loader = DataLoader(DATASET_PATH, train_split=0.7, callbacks={'load': load_cb})
    train_set, test_set = loader.open()
    train_set = train_set.cache().batch(BATCH_SIZE).repeat()
    test_set = test_set.cache().batch(BATCH_SIZE)

    model = ConvModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    model.fit(train_set, epochs=NUM_EPOCHS, validation_data=test_set, steps_per_epoch=10)

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