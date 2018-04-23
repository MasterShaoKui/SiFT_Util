import cv2 as cv
import numpy as np
import config


# roi is 255
def mask_roi(mask):
    shrink_shape = (int(mask.shape[1]/config.mask_scale_factor), int(mask.shape[0]/config.mask_scale_factor))
    mask_min = cv.resize(mask, dsize=shrink_shape, interpolation=cv.INTER_NEAREST)
    mask_final = np.full(shape=(shrink_shape[1], shrink_shape[0]), fill_value=0, dtype=np.uint8)
    for i in range(len(mask_min)):
        for j in range(len(mask_min[i])):
            if np.array_equal(mask_min[i, j], np.array(config.mask_building_color, np.uint8)):
                mask_final[i, j] = 255
    for i in range(len(mask_min)):
        for j in range(len(mask_min[i])):
            if np.array_equal(mask_min[i, j], np.array(config.mask_plate_color, np.uint8)):
                mask_final[i, j] = 255
    mask_final = cv.resize(mask_final, dsize=(mask.shape[1], mask.shape[0]), interpolation=cv.INTER_NEAREST)
    return mask_final


def mask_no_car(mask):
    shrink_shape = (int(mask.shape[1]/config.mask_scale_factor), int(mask.shape[0]/config.mask_scale_factor))
    mask_min = cv.resize(mask, dsize=shrink_shape, interpolation=cv.INTER_NEAREST)
    mask_final = np.full(shape=(shrink_shape[1], shrink_shape[0]), fill_value=255, dtype=np.uint8)
    for i in range(len(mask_min)):
        for j in range(len(mask_min[i])):
            if np.array_equal(mask_min[i, j], np.array(config.mask_car_color, np.uint8)):
                mask_final[i, j] = 0
    mask_final = cv.resize(mask_final, dsize=(mask.shape[1], mask.shape[0]), interpolation=cv.INTER_NEAREST)
    return mask_final

