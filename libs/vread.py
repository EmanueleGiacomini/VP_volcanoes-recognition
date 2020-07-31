"""
Module used to parse VIEW format images taken from `Volcanoes on Venus` dataset.

The image files are in a format called VIEW. This format consists of two files, a binary file
with extension .sdt (the image data) and an ascii file with extension .spr (header information).
"""

import numpy as np
from os.path import isfile


def vread(file: str) -> np.ndarray:
    """
    vread returns the image identified by `file`. file is required to omit the file extension,
    as both .sdt and .spr files will be inferred.
    :param file: file common name.
    :return: image as np.array
    """
    header = file + '.spr'
    if isfile(header) is not True:
        raise Exception('File does not exists')
    with open(header, 'r') as f:
        # Extract metadata
        ndim = int(f.readline())
        if ndim != 2:
            raise Exception('Can handle only 2D images')
        nc = int(f.readline())
        _ = f.readline()
        _ = f.readline()
        nr = int(f.readline())
        _ = f.readline()
        _ = f.readline()
        rtype = int(f.readline())

        precision = None
        if rtype == 0:
            precision = np.uint8
        elif rtype == 2:
            precision = np.int32
        elif rtype == 3:
            precision = np.float32
        elif rtype == 5:
            precision = np.float64
        else:
            raise Exception('Unrecognized data type')

    # Read data
    data = np.fromfile(file + '.sdt', dtype=precision)
    return np.reshape(data, (nr, nc))
