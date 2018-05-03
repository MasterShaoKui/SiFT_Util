import numpy as np
from scipy.optimize import minimize
import cv2 as cv
import config


def para_to_matrix(fx, fy, theta, xc, yc, u0=None, v0=None):
    assert fx != 0, "fx shouldn't be zero! "
    assert fy != 0, "fy shouldn't be zero! "
    if u0 is None:
        u0 = config.u0_cache
    if v0 is None:
        v0 = config.v0_cache
    k = np.eye(4)
    k[0, 0] = fx
    k[1, 1] = fy
    k[2, 0] = u0
    k[2, 1] = v0
    k_inv = np.eye(4)
    k_inv[0, 0] = 1/fx
    k_inv[1, 1] = 1/fy
    k_inv[2, 0] = -u0/fx
    k_inv[2, 1] = -v0/fy
    r = np.eye(4)
    r[0, 0] = np.cos(theta)
    r[0, 1] = np.sin(theta)
    r[1, 0] = -np.sin(theta)
    r[1, 1] = np.cos(theta)
    r[3, 0] = xc
    r[3, 1] = yc
    # final 4 by 4 matrix
    f = k_inv @ r @ k
    f[2] = f[2] + f[3]
    return f[0:3, 0:2].T


def model(x, *args):
    matrix = x.reshape(2, 3)
    x1 = args[0]
    x2 = args[1]
    x2_predi = np.dot(matrix, x1)
    regularize = np.square(matrix[0, 2]) + np.square(matrix[1, 2])
    regularize = regularize / 10
    return np.sum(np.square(x2 - x2_predi)) / x2.shape[0] + regularize


def model_five_para(x, *args):
    matrix = np.hstack((np.array(x[3]), x)).reshape(2, 3)
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
    result = minimize(fun=model, x0=init_matrix, args=(x1.T, x2.T), method='Nelder-Mead', tol=0.0001)
    matrix = result.x.reshape(2, 3)
    # Assume driving straight.
    if matrix[0, 0] < 1.0:
        matrix[0, 0] = 1.0
    if matrix[1, 1] < 1.0:
        matrix[1, 1] = 1.0
    return matrix


def model_affine(x, *args):
    matrix = para_to_matrix(x[0], x[1], x[2], x[3], x[4], config.u0_cache, config.v0_cache)
    x1 = args[0]
    x2 = args[1]
    x2_predi = np.dot(matrix, x1)
    return np.sum(np.square(x2 - x2_predi)) / x2.shape[0]


def calculate_affine_matrix(x1, x2):
    assert x1.shape[0] == x2.shape[0], "Shapes are not equal! "
    if x1.shape[1] == 2:
        x1 = np.hstack((x1, np.ones(shape=(x1.shape[0], 1), dtype=np.float32)))
    # calculate initial matrix
    # fx fy theta xc yc
    init_value = np.array([3600, 3600, 0, 0, 0])
    result = minimize(fun=model_affine, x0=init_value, args=(x1.T, x2.T), method='Nelder-Mead', tol=0.0001)
    x = result.x
    # if np.abs(np.cos(x[2]) - 1) < 0.001:
    #     x[2] = 0.0
    print("result parameters: \r\n", result.x)
    print("result matrix: \r\n", para_to_matrix(x[0], x[1], x[2], x[3], x[4], config.u0_cache, config.v0_cache))
    return para_to_matrix(x[0], x[1], x[2], x[3], x[4], config.u0_cache, config.v0_cache)
