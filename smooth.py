import numpy as np
import os
from scipy.ndimage.filters import gaussian_filter1d


def smooth_frame_matrices_gaussian(matrices):
    assert len(matrices.shape) == 3, "len(matrices.shape) should be 3! "
    for i in range(matrices.shape[1]):
        for j in range(matrices.shape[2]):
            matrices[:, i, j] = gaussian_filter1d(matrices[:, i, j], 1)
    return matrices


def extract_matrices(folder):
    matrices = np.zeros(shape=(0, 2, 3))
    files = os.listdir(folder + "pics/")
    for f_name in files:
        matrix = np.loadtxt(folder + "/" + f_name, dtype=np.float32).reshape(1, 2, 3)
        matrices = np.vstack((matrices, matrix))
    return matrices
