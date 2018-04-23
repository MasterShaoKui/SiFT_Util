import numpy as np
from scipy.optimize import minimize
import cv2 as cv


def model(x, *args):
    matrix = x.reshape(2, 3)
    x1 = args[0]
    x2 = args[1]
    x2_predi = np.dot(matrix, x1)
    return np.sum(np.square(x2 - x2_predi)) / x2.shape[0]


def calculate_perspective_matrix(x1, x2):
    assert x1.shape[0] == x2.shape[0], "Shapes are not equal! "
    if x1.shape[1] == 2:
        x1 = np.hstack((x1, np.ones(shape=(x1.shape[0], 1), dtype=np.float32)))
    # calculate initial matrix
    init_matrix = cv.getAffineTransform(np.float32(x1[0:3, 0:2]), np.float32(x2[0:3, 0:2]))
    print("init_matrix: ", init_matrix)
    result = minimize(fun=model, x0=init_matrix, args=(x1.T, x2.T), method='Nelder-Mead', tol=0.0001)
    print("result_matrix", result.x)
    return result.x.reshape(2, 3)
