import cv2 as cv
import numpy as np
import config
from mask import mask_roi


# note that valid mask color should be (180 120 120).
# (255, 5, 154) is also available, but not recommend.
def dense_optical_flow(img1, img2, mask1, mask2):
    frame1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    frame2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # p1 is the feature points in img1. p2 will be calculated by flow.
    p1 = cv.goodFeaturesToTrack(frame1, maxCorners=2000, qualityLevel=0.3,
                                minDistance=7, mask=mask_roi(mask1), blockSize=7)
    p1 = p1.reshape(-1, 2)
    p2 = np.zeros_like(p1)
    for i, p in enumerate(p1):
        p2[i] = p + flow[int(p[1]), int(p[0])]

    p1_nxt = np.zeros(shape=(0, 2))
    p2_nxt = np.zeros(shape=(0, 2))
    # check if p2 is in corresponding place
    for i, p in enumerate(p1):
        # if point is in correct region, add that point
        print(mask1[int(p[1]), int(p[0])])
        if np.array_equal(mask1[int(p[1]), int(p[0])], np.array(config.mask_building_color, dtype=np.int)) or \
           np.array_equal(mask1[int(p[1]), int(p[0])], np.array((255, 5, 154))):
            if np.array_equal(mask2[int(p2[i][1]),
                                    int(p2[i][0])], np.array(config.mask_building_color, dtype=np.int)) or \
                    np.array_equal(mask2[int(p2[i][1]), int(p2[i][0])], np.array((255, 5, 154))):
                p1_nxt = np.vstack((p1_nxt, p))
                p2_nxt = np.vstack((p2_nxt, p2[i]))
    # check if p2 is in corresponding place
    return np.array(p1_nxt, dtype=np.int), np.array(p2_nxt, dtype=np.int), flow


color = np.random.randint(0, 255, (100, 3))
img1 = cv.imread("./data/65.jpg", cv.IMREAD_UNCHANGED)
img2 = cv.imread("./data/66.jpg", cv.IMREAD_UNCHANGED)
mask1 = cv.imread("./mask/65.jpg", cv.IMREAD_UNCHANGED)
mask2 = cv.imread("./mask/66.jpg", cv.IMREAD_UNCHANGED)
p1_nxt, p2_nxt, flow = dense_optical_flow(img1, img2, mask1, mask2)
for i, point in enumerate(p1_nxt):
    img1 = cv.circle(img1, (int(point[0]), int(point[1])), 5, color[i].tolist(), -1)
mask = np.zeros_like(img1)
for i, (new, old) in enumerate(zip(p2_nxt, p1_nxt)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
    img2 = cv.circle(img2, (a, b), 5, color[i].tolist(), -1)
img2 = cv.add(img2, mask)
cv.imwrite("1.jpg", img1)
cv.imwrite("2.jpg", img2)
print(p1_nxt.shape)
