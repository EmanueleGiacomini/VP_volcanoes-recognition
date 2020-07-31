"""
dataset_handler module.
The DatasetHandler class is used as a gateway for the Venus Volcanoes dataset.
"""

from libs.vread import vread
from libs.improc import sliding_window, augment
from random import sample
from PIL import Image
import numpy as np

VV_PATH_IMGS = 'Images/'
VV_PATH_TABLES = 'Tables/'
VV_PATH_LABELS = 'GroundTruths/'
DATASET_PATH_IMGS = 'dataset/images/'

#ROI_SIZE = 16 # mean of volc. radius (Radius of circle around volcanoes)
BBOX_SIZE = 32
BBOX_SIZE_HALF = int(BBOX_SIZE / 2)
MAGIC_NUMBER = 13
SAMPLE_NON_VOLCANOES = int(1e4)

class DatasetHandler:
    def __init__(self, path, nrange=(1, 135), augment_flag=True):
        imgs_path_lst = [path + VV_PATH_IMGS + 'img' + str(i)
                         for i in range(nrange[0], nrange[1])]

        self.images = []
        for imgpath in imgs_path_lst:
            try:
                self.images.append(vread(imgpath))
            except Exception:
                continue
        # Load labels
        def load_labels(file: str) -> [np.ndarray]:
            data_lst = []
            with open(file, 'r') as f:
                for line in f.readlines():
                    data_lst.append(np.array(list(map(float, line.split()))))
            return data_lst
        labels_path_lst = [path + VV_PATH_LABELS + 'img' + str(i) + '.lxyr'
                           for i in range(nrange[0], nrange[1])]
        labels_lst = [load_labels(i) for i in labels_path_lst]
        self.labels = labels_lst

        def extract_cropped_volcanoes(images_lst, labels) -> ([np.ndarray], [int]):
            """
            Extracts from images_lst list the concatenation of volcanoes cropped images.
            Each output crop has shape (BBOX_SIZE, BBOX_SIZE)
            :param images_lst: list of volcanoes images
            :param labels: list of labels per volcano of each image
            :return: tuple of lists. The first list represent the cropped set of images whereas the second represent the
                relative labels
            """
            data_lst = []
            labels_lst = []
            for i in range(len(images_lst)):
                img = images_lst[i]
                for lab in labels[i]:
                    x_anchor = min(max(int(lab[1]) - BBOX_SIZE_HALF, 0), 1024 - 2 * BBOX_SIZE_HALF)
                    y_anchor = min(max(int(lab[2]) - BBOX_SIZE_HALF, 0), 1024 - 2 * BBOX_SIZE_HALF)
                    crop_volc = img[y_anchor:y_anchor + 2 * BBOX_SIZE_HALF, x_anchor:x_anchor + 2 * BBOX_SIZE_HALF]
                    data_lst.append(crop_volc)
                    labels_lst.append(int(lab[0]))
            return data_lst, labels_lst
        self.volcanoes, self.volcanoes_labels = extract_cropped_volcanoes(self.images, self.labels)
        print(len(self.volcanoes), len(self.volcanoes_labels))
        # Once volcanoes and labels are extracted. Its time to extract non-volcanoes elements
        def extract_non_volcanoes(images_lst, labels) -> ([np.ndarray], [int]):
            """
            Extracts non-volcanoes samples from images_lst list.
            :param images_lst:
            :param labels:
            :return:
            """
            # print callback
            print_str = '='
            idx_ctr = 0
            data_lst = []
            for image_idx, image in enumerate(images_lst):
                # Sample from grid space of dimension BBOX_SIZE
                image_slides = sliding_window(image, BBOX_SIZE, MAGIC_NUMBER, labels[image_idx])
                data_lst.extend(image_slides)
                idx_ctr += 1
                print('\r' + '=' * idx_ctr + '>', end='')
            labels_lst = [0 for _ in range(len(data_lst))]
            return np.array(data_lst), labels_lst
        self.non_volcanoes, self.non_volcanoes_labels = extract_non_volcanoes(self.images, self.labels)
        # Sample NON_VOLCANOES_SAMPLE elements from self.non_volcanoes
        self.non_volcanoes = sample(self.non_volcanoes, SAMPLE_NON_VOLCANOES)
        self.non_volcanoes_labels = [0 for _ in range(len(self.non_volcanoes))]
        # TODO (perchè andrea vuole prima efficientare il dataset

        """
        print(f'Original Volcanoes images: {len(self.volcanoes)}')
        augmented_images_lst = []
        for i, image in enumerate(self.volcanoes):
            augmented_img, augmented_label = augment(image, self.volcanoes_labels[i])
            augmented_images_lst.append(augmented_img.numpy())
            self.volcanoes_labels.append(augmented_label)
        self.volcanoes.extend(augmented_images_lst)
        print(f'Augmented Volcanoes images: {len(self.volcanoes)}')
        """


        """
        # Crop volcanoes.
        bb_mean = 0.0
        _ctr = 0
        for i in labels_lst:
            for j in i:
                bb_mean += j[3]
                _ctr += 1
        bb_mean /= _ctr
        print(f'Mean bbox size: {int(bb_mean) + 1}')
        """
        """
        maxval = 0
        for i in labels_lst:
            for j in i:
                if j[3] > maxval:
                    maxval = j[3]
        print(maxval)
        """







from matplotlib import pyplot as plt

def plot_volcanoes(img, labels, ax):
    ax.imshow(img, cmap='gray')
    for lab in labels:
        color = ''
        clab = int(lab[0])
        if clab == 1:
            color = 'r'
        elif clab == 2:
            color = 'orange'
        elif clab == 3:
            color = 'yellow'
        elif clab == 4:
            color = 'blue'
        #ax.add_artist(plt.Circle((lab[1], lab[2]), lab[3], color=color, fill=False))
        ax.add_artist(plt.Rectangle((lab[1] - BBOX_SIZE_HALF, lab[2] - BBOX_SIZE_HALF), 2 * BBOX_SIZE_HALF, 2 * BBOX_SIZE_HALF, color=color, fill=False))
        #ax.add_artist(plt.Rectangle((lab[1] - 44, lab[2] - 44), 88, 88, color=color, fill=False))

if __name__ == '__main__':
    data = DatasetHandler('./venus_volcanoes/package/')
    ax_labels = ['dummy', 'definitely', 'probably', 'possibly', 'only a pit']

    fig, axs = plt.subplots(1, 8, sharex=True, sharey=True)
    axs[0].imshow(data.volcanoes[0], cmap='gray')
    for i in range(1, 8):
        augmented_img, _ = augment(data.volcanoes[0], 1)
        augmented_img = augmented_img.numpy()
        axs[i].imshow(np.reshape(augmented_img, (augmented_img.shape[0], augmented_img.shape[1])), cmap='gray')
    plt.show()

    """
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
    for i in range(1):
        for j in range(1):
            plot_volcanoes(data.images[i*5 + j], data.labels[i*5 + j], axs)
            #axs[i][j].imshow(data.volcanoes[600 + i*5 + j], cmap='gray')
            #axs[i][j].set_title(ax_labels[data.volcanoes_labels[600 + i*5 + j]])
    plt.show()
    """