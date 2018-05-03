import numpy as np
import config


def get_bev_base_points_origin(bev_img):
    margin_w = 50
    margin_h_up = 50
    margin_h_down = 50
    step = 100
    bev_w = bev_img.shape[1]
    bev_h = bev_img.shape[0]
    while np.array_equal(bev_img[bev_h - margin_h_down, margin_w], np.array([0, 0, 0])):
        margin_h_down += step
    while np.array_equal(bev_img[bev_h - margin_h_down, bev_w - margin_w], np.array([0, 0, 0], dtype=np.int)):
        margin_h_down += step
    middle = (bev_h - margin_h_up - margin_h_down)/2 + margin_h_up
    middle = int(middle)
    points = [[margin_h_up, margin_w], [margin_h_up, int(bev_w / 2)], [margin_h_up, bev_w - margin_w],
              [middle, margin_w], [middle, int(bev_w / 2)], [middle, bev_w - margin_w],
              [bev_h - margin_h_down, margin_w], [bev_h - margin_h_down, int(bev_w / 2)],
              [bev_h - margin_h_down, bev_w - margin_w]]
    points = np.flip(points, 1)
    return np.array(points)


def get_bev_base_points_origin_list(bev_img):
    margin_w = 50
    margin_h_up = 50
    margin_h_down = 50
    step = 100
    bev_w = bev_img.shape[1]
    bev_h = bev_img.shape[0]
    while np.array_equal(bev_img[bev_h - margin_h_down, margin_w], np.array([0, 0, 0])):
        margin_h_down += step
    while np.array_equal(bev_img[bev_h - margin_h_down, bev_w - margin_w], np.array([0, 0, 0], dtype=np.int)):
        margin_h_down += step
    middle = (bev_h - margin_h_up - margin_h_down)/2 + margin_h_up
    middle = int(middle)
    points = [[margin_h_up, margin_w], [margin_h_up, int(bev_w / 2)], [margin_h_up, bev_w - margin_w],
              [middle, margin_w], [middle, int(bev_w / 2)], [middle, bev_w - margin_w],
              [bev_h - margin_h_down, margin_w], [bev_h - margin_h_down, int(bev_w / 2)],
              [bev_h - margin_h_down, bev_w - margin_w]]
    return points


def get_bev_base_points_middle_only(bev_img):
    margin_w = 50
    margin_h_up = 50
    margin_h_down = 50
    step = 100
    bev_w = bev_img.shape[1]
    bev_h = bev_img.shape[0]
    while np.array_equal(bev_img[bev_h - margin_h_down, margin_w], np.array([0, 0, 0])):
        margin_h_down += step
    while np.array_equal(bev_img[bev_h - margin_h_down, bev_w - margin_w], np.array([0, 0, 0], dtype=np.int)):
        margin_h_down += step
    middle = (bev_h - margin_h_up - margin_h_down)/2 + margin_h_up
    middle = int(middle)
    points = [[margin_h_up, int(bev_w / 2)], [middle, int(bev_w / 2)], [bev_h - margin_h_down, int(bev_w / 2)]]
    points = np.flip(points, 1)
    return np.array(points)


def get_bev_base_points(bev_img):
    margin_h_up = 50
    margin_h_down = 50
    bev_w = bev_img.shape[1]
    bev_h = bev_img.shape[0]
    middle = (bev_h - margin_h_up - margin_h_down)/2 + margin_h_up
    middle = int(middle)
    middle_h = int(bev_w / 2)
    points = list([[margin_h_up, middle_h],
                   [int((margin_h_up + middle)/2), middle_h],
                   [middle, middle_h]])
    step = (bev_h - margin_h_down - middle) / (config.down_points_num + 1)
    for i in range(config.down_points_num):
        points.append([middle + int(step * (i+1)), middle_h])
    points.append([bev_h - margin_h_down, middle_h])
    points += get_bev_base_points_origin_list(bev_img)
    points += get_bev_base_points_origin_list(bev_img)
    points = np.flip(points, 1)
    return np.array(points)


def norm_cv_transform(dst):
    dst = dst[0]
    dst = dst / dst[:, 2].reshape(dst.shape[0], 1)
    dst = dst[:, 0:2]
    return dst
