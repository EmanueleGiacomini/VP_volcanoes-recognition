"""
Image Processing Library
"""

import numpy as np
from random import random
import tensorflow as tf


def sliding_window(image: np.ndarray, wsize: int, stride: int, filter_lst: [np.ndarray] = None):
    """
    Returns the sliding window over a given ndarray 'image'
    :param image:
    :param wsize:
    :param stride:
    :param filter_lst:
    :return:
    """

    def slide_filter():
        def boxCircleIntersect(box: np.ndarray, circle: np.ndarray) -> bool:
            """

            :param box: [xanch, yanch, wsize]
            :param circle: [_, xcnt, ycnt, radius]
            :return:
            """
            p1, p2, p3, p4 = np.array([box[0], box[1]]), \
                             np.array([box[0] + wsize, box[1]]), \
                             np.array([box[0], box[1] + wsize]), \
                             np.array([box[0] + wsize, box[1] + wsize]),
            ccenter = np.array([circle[1], circle[2]])
            return circle[3] >= np.sum(np.power(p1 - ccenter, 2)) or \
                   circle[3] >= np.sum(np.power(p2 - ccenter, 2)) or \
                   circle[3] >= np.sum(np.power(p3 - ccenter, 2)) or \
                   circle[3] >= np.sum(np.power(p4 - ccenter, 2))

        window_lst = []
        width, height = image.shape
        for i in range(0, height - wsize, stride):
            for j in range(0, width - wsize, stride):
                window = image[i:i + wsize, j:j + wsize]
                invalid_flag = False
                for f in filter_lst:
                    if boxCircleIntersect(np.array([j, i]), f) or \
                            np.count_nonzero(window) < window.shape[0] * window.shape[1]:
                        invalid_flag = True
                        break
                if invalid_flag:
                    continue
                else:
                    window_lst.append(window)
        return window_lst

    def slide_raw():
        window_lst = []
        width, height = image.shape
        for i in range(0, height - wsize, stride):
            for j in range(0, width - wsize, stride):
                window_lst.append(image[i:i + wsize, j:j + wsize])
        return window_lst

    if filter_lst is None or len(filter_lst) == 0:
        slide_fn = slide_raw
    else:
        slide_fn = slide_filter

    return slide_fn()


def convert(image: np.ndarray, label) -> tf.Tensor:
    image = tf.image.convert_image_dtype(image[..., np.newaxis], tf.float64)
    return image, label


def augment(image: np.ndarray, label: int):
    #image, label = convert(image, label)
    # Apply Mirror
    image = tf.image.flip_left_right(image)
    image = tf.image.flip_up_down(image)
    # Rotate
    image = tf.image.rot90(image)
    return image, label

def augmentImage(image: tf.Tensor):
    image = tf.image.flip_left_right(image)
    image = tf.image.flip_up_down(image)
    image = tf.image.rot90(image)
    return image

def augmentImages(images: [tf.Tensor], labels: [int], labels_filter=[0]) -> ([tf.Tensor], [int]):
    images_augmented = []
    labels_augmented = []
    for i, image in enumerate(images):
        if labels[i] not in labels_filter:
            image_augm, label_augm = augmentImage(image, labels[i])
            images_augmented.append(image_augm)
            labels_augmented.append(label_augm)
    images_augmented.extend(images)
    labels_augmented.extend(labels)
    return images_augmented, labels_augmented
