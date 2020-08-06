"""
plotter.py

"""

from libs.dloader import DataLoader
from libs.callbacks import load_cb
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

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
    ax.imshow(image[:,:,0], cmap='gray')
    label_str = LABELS_DECODE[np.argmax(label)]
    if percentage is None:
        ax.set_title(label_str)
    else:
        ax.set_title(f'{label_str} ({percentage * 100})')
    return

LABELS_DECODE = [
    'Not volcano',
    'Definitely volcano',
    'Probably volcano',
    'Possibly volcano',
    'Only a pit'
]

if __name__ == '__main__':
    tf.keras.backend.set_floatx('float64')

    loader = DataLoader(DATASET_PATH, train_split=0.7, callbacks={'load': load_cb})
    train_set, test_set = loader.open()
    # Plot histogram (training, validation)
    fig, axs = plt.subplots(2, 1, sharex=True)
    plot_hist(axs[0], train_set)
    axs[0].set_title('Training distribution')
    plot_hist(axs[1], test_set)
    axs[1].set_title('Validation distribution')
    plt.savefig('./images/histogram.jpg')
    # Plot samples
    plt.rcParams.update({'font.size': 12})
    fig, axs = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(20,20))
    i, j = 0, 0
    for image, label in train_set.shuffle(100).take(25):
        print(np.argmax(label))
        plot_sample(axs[i][j], image, label, 1.)
        j += 1
        if j % 5 == 0 and j != 0:
            i += 1
            j = 0
    plt.savefig('./images/training_sample.jpg')
    # Test samples
    fig, axs = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(20, 20))
    i, j = 0, 0
    for image, label in test_set.shuffle(100).take(25):
        plot_sample(axs[i][j], image, label)
        j += 1
        if j % 5 == 0 and j != 0:
            i += 1
            j = 0
    plt.savefig('./images/test_sample.jpg')


