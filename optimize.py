import numpy as np
from scipy.optimize import minimize
import cv2 as cv


def p_m_form(bits):
    return np.array([[bits[0], bits[1], bits[2]],
                     [bits[3], bits[4], bits[5]],
                     [bits[6], bits[7], 1.0]],
                    dtype=np.float64)


def model(x, *args):
    matrix = p_m_form(x)
    x1 = args[0]
    x2 = args[1]
    x2_predi = np.dot(matrix, x1)
    x2_predi = x2_predi / x2_predi[2, :].reshape(1, x2_predi.shape[1])
    return np.sum(np.square(x2[:2, :] - x2_predi[:2, :])) / x2.shape[0]


def calculate_perspective_matrix(x1, x2):
    assert x1.shape[0] == x2.shape[0], "Shapes are not equal! "
    if x1.shape[1] == 2:
        x1 = np.hstack((x1, np.ones(shape=(x1.shape[0], 1), dtype=np.float32)))
    if x2.shape[1] == 2:
        x2 = np.hstack((x2, np.ones(shape=(x2.shape[0], 1), dtype=np.float32)))
    # calculate initial matrix
    init_matrix = cv.getPerspectiveTransform(np.float32(x1[0:4, 0:2]), np.float32(x2[0:4, 0:2]))
    print(init_matrix, "\r\n")
    result = minimize(fun=model, x0=init_matrix.flat[0:8], args=(x1.T, x2.T), method='Nelder-Mead', tol=0.00001)
    # print(minimize(fun=model, x0=np.random.randint(low=-10, high=10, size=8),
    #                args=(x1.T, x2.T), method='Nelder-Mead',
    #                tol=0.00001).x)
    return p_m_form(result.x)
