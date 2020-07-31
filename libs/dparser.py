"""
dloader module.
DataParser class is used to parse images and labels from venus_volcanoes dataset.
"""

from libs.vread import vread
import numpy as np

VV_PATH_IMGS = 'Images/'
VV_PATH_TABLES = 'Tables/'
VV_PATH_LABELS = 'GroundTruths/'
DATASET_PATH_IMGS = 'dataset/images/'


def load_metadata(file: str) -> [np.ndarray]:
    data = []
    with open(file, 'r') as f:
        for line in f.readlines():
            data.append(np.array(list(map(float, line.split()))))
    return data


class DataParser:
    def __init__(self, path, nrange=(1, 135)):
        images_path_lst = [path + VV_PATH_IMGS + 'img' + str(i)
                           for i in range(nrange[0], nrange[1])]
        self.images = []
        # Extract Images
        for i in images_path_lst:
            try:
                self.images.append(vread(i))
            except Exception:
                continue

        # Extract Labels
        labels_path_lst = [path + VV_PATH_LABELS + 'img' + str(i) + '.lxyr'
                           for i in range(nrange[0], nrange[1])]
        self.metadata = [load_metadata(i) for i in labels_path_lst]

    def getImages(self) -> [np.ndarray]:
        return self.images

    def getMetaData(self):
        return self.metadata
