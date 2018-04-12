import cv2 as cv
import numpy as np


def get_bev_base_points(bev_img_dir):
    margin_w = 50
    margin_h_up = 50
    margin_h_down = 50
    step = 100
    bev_img = cv.imread(bev_img_dir, cv.IMREAD_UNCHANGED)
    bev_w = bev_img.shape[1]
    bev_h = bev_img.shape[0]
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
    return points
