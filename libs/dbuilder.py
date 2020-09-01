"""
dbuilder module
The DataBuilder is used to generate training/test samples from the
previously loaded images
"""

from libs.dparser import DataParser
from libs.improc import sliding_window
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def extract_volcanoes(images: [tf.Tensor], metadata: [np.ndarray], bbox_size: (int, int)) -> ([tf.Tensor], [int]):
    X_IDX = 1
    Y_IDX = 2
    bbox_size_half = (int(bbox_size[0] / 2), int(bbox_size[1] / 2))

    data = []
    labels = []
    for i in range(len(images)):
        image = images[i]
        # Bound values for each cordinate
        x_min, x_max = 0, 1024
        y_min, y_max = x_min, x_max

        for meta in metadata[i]:
            if len(meta) == 0:
                continue
            x_anchor = min(max(int(meta[X_IDX]) - bbox_size_half[0], x_min), x_max - bbox_size[0])
            y_anchor = min(max(int(meta[Y_IDX]) - bbox_size_half[1], y_min), y_max - bbox_size[1])
            data.append(image[y_anchor:y_anchor + bbox_size[1], x_anchor:x_anchor + bbox_size[0]])
            labels.append(int(meta[0]))
    return data, labels


def sample_non_volcanoes(images: [tf.Tensor], metadata: [np.ndarray], num_samples: int,
                         bbox_size: (int, int), stride_size: int = 15) -> ([tf.Tensor], [int]):
    def loading_bar_cb(cval, fval):
        if cval == fval:
            print(' Completed.')
        else:
            ratio = int((cval / fval) * 30)
            print('\r' + '|' + '=' * ratio + '>' + ' ' * (30 - ratio) + '|', end='')

    # Generate all possible crops
    crops = []
    for i, image in enumerate(images):
        # Sample from grid space of dimension bbox_size
        crops.extend(sliding_window(image, bbox_size[0], stride_size, metadata[i]))
        loading_bar_cb(i, len(images) - 1)
    print(f'Generated {len(crops)} crops.')
    # Sample from crops
    samples = random.sample(crops, num_samples)
    return samples, [0 for _ in range(len(samples))]


class DataBuilder:
    def __init__(self, path: str, bbox_size=64, non_volcanoes_samples=int(3e3), stride=15):
        # Load data from a DataParser element
        dloader = DataParser(path)
        images_raw = dloader.getImages()
        images_tf = list(map(lambda x:
                             tf.convert_to_tensor(x, tf.float64), images_raw))
        metadata_raw = dloader.getMetaData()
        # Volcanoes images/labels
        vimages, vlabels = extract_volcanoes(images_tf, metadata_raw, bbox_size=(bbox_size, bbox_size))
        nvimages, nvlabels = sample_non_volcanoes(random.sample(images_tf, 4),
                                                  metadata_raw,
                                                  non_volcanoes_samples,
                                                  bbox_size=(bbox_size, bbox_size),
                                                  stride_size=stride)
        images = vimages + nvimages
        labels = vlabels + nvlabels
        # Shuffle images and labels
        c = list(zip(images, labels))
        random.shuffle(c)
        self.images, self.labels = zip(*c)

    def store(self, path: str, labels:['str'], train_ratio=0.7):
        """
        Stores the computed data into the following format:
        -path/labels.csv [the generic row i represent the label of i.jpg]
        -path/i.jpg      [i is a generic number between [0, no.volcanoes + no.not_volcanoes]
        :param train_ratio:
        :param path:
        :return:
        """

        # Split train and test images
        X_train, X_test, y_train, y_test = train_test_split(self.images, self.labels, test_size=(1. - train_ratio),
                                                            random_state=42)

        def storeImage(image: tf.Tensor, label: str, current_path: str, filename: str) -> None:
            image_encoded = tf.cast(image, tf.uint8)
            image_encoded = tf.expand_dims(image_encoded, -1)
            image_encoded = tf.image.encode_jpeg(image_encoded, quality=100, format='grayscale')
            impath = current_path + label + '/' + filename + '.jpg'
            tf.io.write_file(impath, image_encoded)
            return

        labels_counter = [0 for _ in range(len(labels))]
        current_path = path + 'training_data/'
        for image, label in zip(X_train, y_train):
            filename = labels[label] + '_image_' + str(labels_counter[label])
            labels_counter[label] += 1
            storeImage(image, labels[label], current_path, filename)

        labels_counter = [0 for _ in range(len(labels))]
        current_path = path + 'validation_data/'
        for image, label in zip(X_test, y_test):
            filename = labels[label] + '_image_' + str(labels_counter[label])
            labels_counter[label] += 1
            storeImage(image, labels[label], current_path, filename)
        """
        # Store images
        for i, image in enumerate(self.images):
            image_encoded = tf.cast(image, tf.uint8)
            image_encoded = tf.expand_dims(image_encoded, -1)
            image_encoded = tf.image.encode_jpeg(image_encoded, quality=100, format='grayscale')
            tf.io.write_file(path + str(i) + '.jpg', image_encoded)
        with open(path + 'labels.csv', 'w') as f:
            for label in self.labels:
                f.write(str(label))
                f.write('\n')
        """


from matplotlib import pyplot as plt

if __name__ == '__main__':
    labels = ['not_present', 'definitely',
              'probably', 'possibly',
              'only_pit']
    dbuilder = DataBuilder('./venus_volcanoes/package/')
    dbuilder.store('./dataset/', labels, train_ratio=0.7)
    fig, axs = plt.subplots(2, 10, sharex=True, sharey=True)
    for i in range(2):
        for j in range(10):
            axs[i][j].imshow(dbuilder.images[10 * i + j], cmap='gray')
            axs[i][j].set_title(labels[dbuilder.labels[10 * i + j]])
    plt.show()
