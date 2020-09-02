"""
plotter.py

"""

from libs.dloader import DataLoader
from libs.callbacks import load_cb
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import io

DATASET_PATH = './dataset/'
NUM_CLASSES = 5


def plot_hist(ax, data: tf.data.Dataset, num_classes: int = 5):
    labels_lst = []
    # Traverse the histogram
    for _, label_enc in data:
        labels_lst.append(np.argmax(label_enc))
    ax.hist(labels_lst, bins=num_classes)
    return


def plot_sample(ax, image, label, percentage=None):
    ax.imshow(image[:, :, 0], cmap='gray')
    label_str = LABELS_DECODE[np.argmax(label)]
    if percentage is None:
        ax.set_title(label_str)
    else:
        ax.set_title(f'{label_str} ({percentage * 100})')
    return


def plot_sample_grid(data: tf.data.Dataset, labels: ['str'], figshape=(5, 5)):
    num_samples = figshape[0] * figshape[1]
    figure = plt.figure(figsize=(15, 15))
    i = 1
    for image, label_onehot in data.unbatch().take(num_samples):
        plt.subplot(figshape[0], figshape[1], i, title=labels[np.argmax(label_onehot)])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap='gray')
        i += 1
    return figure


def plot_predictions_grid(data: tf.data.Dataset, model: tf.keras.Model, labels: ['str'], figshape=(5, 5)):
    num_samples = figshape[0] * figshape[1]
    figure = plt.figure(figsize=(10, 10))
    i = 1
    batch = data.take(1)
    y_pred_vect = model.predict(batch)
    batch = batch.unbatch()
    for image, y_true in batch:
        if i > num_samples:
            break
        y_pred = y_pred_vect[i]
        y_pred_max = np.max(y_pred)
        y_pred_idx = np.argmax(y_pred)
        plt.subplot(figshape[0], figshape[1], i, title=f'{labels[y_pred_idx]} ({y_pred_max:.2f})\n'
                                                       f'[{labels[np.argmax(y_true)]}]')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap='gray')
        i += 1
    return figure


def figure_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=1)
    image = tf.expand_dims(image, 0)
    return image


LABELS_DECODE = [
    'Not volcano',
    'Definitely volcano',
    'Probably volcano',
    'Possibly volcano',
    'Only a pit'
]

if __name__ == '__main__':
    tf.keras.backend.set_floatx('float64')

    loader = DataLoader(DATASET_PATH, augment=True, callbacks={'load': load_cb})
    train_set, test_set = loader.open()
    # Plot histogram (training, validation)

    fig, axs = plt.subplots(2, 1, sharex=True)
    plot_hist(axs[0], train_set)
    axs[0].set_title('Training distribution')
    plot_hist(axs[1], test_set)
    axs[1].set_title('Validation distribution')
    plt.savefig('./images/histogram.png')
    # Plot samples
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(20, 20))
    i, j = 0, 0
    for image, label in train_set.shuffle(100).take(25):
        plot_sample(axs[i][j], image, label, 1.)
        j += 1
        if j % 5 == 0 and j != 0:
            i += 1
            j = 0
    plt.savefig('./images/training_sample.png')
    # Test samples
    fig, axs = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(20, 20))
    i, j = 0, 0
    for image, label in test_set.shuffle(100).take(25):
        plot_sample(axs[i][j], image, label)
        j += 1
        if j % 5 == 0 and j != 0:
            i += 1
            j = 0
    plt.savefig('./images/test_sample.png')
