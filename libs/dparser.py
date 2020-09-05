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

from plotter import plot_full_volcanoes
from matplotlib import pyplot as plt

if __name__ == '__main__':
    dp = DataParser('./venus_volcanoes/package/')
    samples_ctr = 0
    for i in range(len(dp.metadata)):
        samples_ctr += len(dp.metadata[i])
    print(f'Total of {samples_ctr} samples.')
    """fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(dp.images[0], cmap='gray')
    ax.set_title('Venus Surface SAR readings')
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.savefig('./images/venus_surface_raw.png')"""
    """figure = plot_full_volcanoes(dp.images[15], dp.metadata[15])
    ax = figure.axes[0]
    plt.rcParams.update({'font.size': 24})
    ax.set_title('Venus surface image with volcanoes highlight')
    plt.savefig('./images/venus_surface.png')
    """
    exit(0)