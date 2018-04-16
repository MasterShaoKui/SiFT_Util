import cv2 as cv
import numpy as np
import config
mask = cv.imread("./mask/65.jpg", cv.IMREAD_UNCHANGED)
shrink_shape = (int(mask.shape[1]/25), int(mask.shape[0]/25))
mask_min = cv.resize(mask, dsize=shrink_shape, interpolation=cv.INTER_NEAREST)
mask_final = np.full(shape=(shrink_shape[1], shrink_shape[0]), fill_value=255, dtype=np.intc)
for i in range(len(mask_min)):
    for j in range(len(mask_min[i])):
        if np.array_equal(mask_min[i, j], np.array(config.mask_building_color, np.intc)):
            mask_final[i, j] = 0
for i in range(len(mask_min)):
    for j in range(len(mask_min[i])):
        if np.array_equal(mask_min[i, j], np.array(config.mask_plate_color, np.intc)):
            mask_final[i, j] = 0
mask_final = cv.resize(mask_final, dsize=(mask.shape[1], mask.shape[0]), interpolation=cv.INTER_NEAREST)
cv.imwrite("pic.jpg", mask_final)
print(mask_final)

