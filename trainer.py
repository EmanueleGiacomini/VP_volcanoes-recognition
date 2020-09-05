"""
Training script for the model.
"""

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import os
import datetime

from libs.dloader import DataLoader
from libs.conv_model import ConvModel, createConvModel2
from libs.callbacks import load_cb
from libs.tf_callbacks import LearningRateLoggingCallback
from plotter import LABELS_DECODE, plot_sample_grid, figure_to_image, plot_predictions_grid

DATASET_PATH = './dataset/'
NUM_EPOCHS = 400
BATCH_SIZE = 128
NUM_CLASSES = 5


def plot_confusion_matrix(confusion_mat: np.array):
    fig, ax = plt.subplots(figsize=(5, 5))
    # Normalize confusion matrix row-wise
    cmax_sums = confusion_mat.sum(axis=1)
    confusion_mat = confusion_mat / cmax_sums[:, np.newaxis]
    df_cm = pd.DataFrame(confusion_mat, range(NUM_CLASSES), range(NUM_CLASSES))
    ax = sn.heatmap(df_cm, annot=True, fmt='.1f')
    ax.set_ylim(len(confusion_mat), 0.0)
    return fig


def compute_class_weights(train_set: tf.data.Dataset, class_percentage: [int]) -> {}:
    n_samples = tf.data.experimental.cardinality(train_set).numpy()
    weights = [n_samples / (NUM_CLASSES * (class_percentage[i] * n_samples)) for i in range(NUM_CLASSES)]
    class_weights = {}
    for i in range(len(weights)):
        class_weights[i] = weights[i]
    return class_weights


if __name__ == '__main__':
    print(tf.__version__)
    # Setup GPU usage for TensorFlow, if available
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    # tf.debugging.set_log_device_placement(True)
    tf.keras.backend.set_floatx('float64')
    # Setup random seed
    tf.random.set_seed(42)
    np.random.seed(42)

    loader = DataLoader(DATASET_PATH, augment=True, callbacks={'load': load_cb})
    train_set, test_set = loader.open()
    train_set = train_set.batch(BATCH_SIZE).cache().repeat()
    test_set = test_set.batch(BATCH_SIZE)


    optimizer_lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,
                                                                        decay_steps=50,
                                                                        decay_rate=1,
                                                                        staircase=False)
    model_optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer_lr_decay,
                                               beta_1=0.9,
                                               beta_2=0.999,
                                               epsilon=1e-07)
    model = ConvModel()
    # model = createConvModel2()
    #model.compile(optimizer=model_optimizer, loss='categorical_crossentropy', metrics=['acc'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    #model.compile(optimizer='sgd', loss='kullback_leibler_divergence', metrics=['acc'])
    #model.load_weights('training/cp-100.ckpt')


    class_weights = compute_class_weights(train_set, loader.class_percentage)
    print(class_weights)

    cb_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=10,
                                             restore_best_weights=True,
                                             verbose=True)
    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Learning Rate Update Callback
    cb_lr = LearningRateLoggingCallback()

    file_writer = tf.summary.create_file_writer(log_dir)

    with file_writer.as_default():
        figure = plot_sample_grid(train_set, LABELS_DECODE, (5, 5))
        tf.summary.image('Training data', figure_to_image(figure), step=0)
        figure = plot_sample_grid(test_set, LABELS_DECODE, (5, 5))
        tf.summary.image('Validation data', figure_to_image(figure), step=0)

    checkpoint_path = 'training/cp-{epoch:03d}.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cb_mckpt = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=True,
                                                  period=10)

    training_history = model.fit(train_set, epochs=NUM_EPOCHS, validation_data=test_set,
                                 steps_per_epoch=15, class_weight=class_weights,
                                 callbacks=[cb_es, cb_mckpt, cb_tensorboard])
    # Fill confusion matrix
    cm = np.zeros(shape=(NUM_CLASSES, NUM_CLASSES))
    for x_val, y_val in test_set.take(10):
        y_true = np.argmax(y_val, axis=1)
        y_pred = np.argmax(model.predict(x_val), axis=1)
        tcm = confusion_matrix(y_true, y_pred, labels=range(0, NUM_CLASSES))
        cm += tcm

    figure = plot_confusion_matrix(cm)
    plt.tight_layout()
    with file_writer.as_default():
        tf.summary.image('Classifier confusion matrix', figure_to_image(figure), step=0)

    # Image with predictions
    figure = plot_predictions_grid(test_set, model, LABELS_DECODE)
    plt.tight_layout()
    with file_writer.as_default():
        tf.summary.image('Test set predictions', figure_to_image(figure), step=0)
    exit(0)
