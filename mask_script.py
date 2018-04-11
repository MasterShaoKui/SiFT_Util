import cv2 as cv
import numpy as np
import config
from config import mask_margin as margin
img = cv.imread("./mask/65.jpg", cv.IMREAD_UNCHANGED)
mask = np.zeros(shape=(img.shape[0], img.shape[1]))
ban_list = [np.array(config.mask_car_color), np.array(config.mask_human_color)]
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if np.array_equal(img[i, j], np.array(config.mask_car_color, dtype=np.int)):
            mask[i, j] = 1
            if i - margin > 0 and i + margin < mask.shape[0] and j - margin > 0 and j + margin < mask.shape[1]:
                mask[i-margin:i+margin, j-margin:j+margin] = 1
        print(i, j)
cv.imwrite("pic.jpg", mask)
