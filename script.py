import cv2 as cv
# a = cv.imread("./data/65.jpg", cv.IMREAD_UNCHANGED)
# b = cv.imread("./data/66.jpg", cv.IMREAD_UNCHANGED)
# cv.imwrite("./data/65-66.jpg", a*0.5+b*0.5)

import numpy as np
bev_matrix = np.loadtxt("./t173_pm/65.txt", dtype=np.float64)
bev_matrix_inv = np.linalg.inv(bev_matrix)
bev_img = cv.imread("./bev_pics/65.jpg", cv.IMREAD_UNCHANGED)
bev_w = bev_img.shape[1]
bev_h = bev_img.shape[0]
margin_w = 50
margin_h_up = 50
margin_h_down = 50
step = int(bev_h / 10)
while np.array_equal(bev_img[bev_h - margin_h_down, margin_w], np.array([0, 0, 0])):
    margin_h_down += step
while np.array_equal(bev_img[bev_h - margin_h_down, bev_w - margin_w], np.array([0, 0, 0], dtype=np.int)):
    margin_h_down += step
middle = (bev_h - margin_h_up - margin_h_down)/2 + margin_h_up
middle = int(middle)
# points = [[margin_w, margin_h_up], [int(bev_w / 2), margin_h_up], [bev_w - margin_w, margin_h_up],
#           [margin_w, middle], [int(bev_w/2), middle], [bev_w - margin_w, middle],
#           [margin_w, margin_h_down], [int(bev_w/2), margin_h_down], [bev_w - margin_w, margin_h_down]]
points = [[margin_h_up, margin_w], [margin_h_up, int(bev_w / 2)], [margin_h_up, bev_w - margin_w],
          [middle, margin_w], [middle, int(bev_w / 2)], [middle, bev_w - margin_w],
          [bev_h - margin_h_down, margin_w], [bev_h - margin_h_down, int(bev_w / 2)],
          [bev_h - margin_h_down, bev_w - margin_w]]
for p in points:
    cv.circle(bev_img, (int(p[1]), int(p[0])), 5, (0, 0, 255), 6)
cv.imwrite("pic.jpg", bev_img)
