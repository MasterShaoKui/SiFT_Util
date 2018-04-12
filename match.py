import numpy as np
import cv2 as cv
import config
from config import mask_margin as margin


def match_standard(des1, des2):
    matches = list()
    des1 = des2 * 2 + des1
    return matches


def refine_match_moving(matches, kp1, kp2, center):
    i = 0
    while i < len(matches):
        m = matches[i]
        end1 = np.array(kp1[m.queryIdx].pt)
        end2 = np.array(kp2[m.trainIdx].pt)
        if end1[0] < center[0] and end1[1] < center[1]:
            if end2[0] > end1[0] or end2[1] > end1[1]:
                del matches[i]
                i -= 1
        elif end1[0] > center[0] and end1[1] < center[1]:
            if end2[0] < end1[0] or end2[1] > end1[1]:
                del matches[i]
                i -= 1
        elif end1[0] < center[0] and end1[1] > center[1]:
            if end2[0] > end1[0] or end2[1] < end1[1]:
                del matches[i]
                i -= 1
        elif end1[0] > center[0] and end1[1] > center[1]:
            if end2[0] < end1[0] or end2[1] < end1[1]:
                del matches[i]
                i -= 1
        i += 1


def refine_match_without_car(matches, kp1, kp2):
    i = 0
    while i < len(matches):
        m = matches[i]
        end1 = np.array(kp1[m.queryIdx].pt)
        end2 = np.array(kp2[m.trainIdx].pt)
        if end1[1] > 800:
            del matches[i]
            i -= 1
        i += 1


def refine_match_mask_filter(matches, kp1, kp2, mask_dir1, mask_dir2):
    mask1 = cv.imread(mask_dir1, cv.IMREAD_UNCHANGED)
    mask2 = cv.imread(mask_dir2, cv.IMREAD_UNCHANGED)
    i = 0
    while i < len(matches):
        m = matches[i]
        end1 = np.array(kp1[m.queryIdx].pt, dtype=np.int)
        # 不要过于边缘的点
        if end1[0] - margin <= 0 or end1[0] + margin >= mask1.shape[0] \
                or end1[1] - margin <= 0 or end1[1] + margin >= mask1.shape[1]:
            del matches[i]
            i -= 1
            continue
        bool_map = mask1[end1[0]-margin:end1[0]+margin, end1[1]-margin:end1[1]+margin] == \
            np.array(config.mask_car_color, dtype=np.int)
        if True in bool_map.flat:
            del matches[i]
            i -= 1
            continue
        i += 1
    i = 0
    while i < len(matches):
        m = matches[i]
        end2 = np.array(kp2[m.trainIdx].pt, dtype=np.int)
        # 不要过于边缘的点
        if end2[0] - margin <= 0 or end2[0] + margin >= mask2.shape[0] \
                or end2[1] - margin <= 0 or end2[1] + margin >= mask2.shape[1]:
            del matches[i]
            continue
        bool_map = mask2[end2[0]-margin:end2[0]+margin, end2[1]-margin:end2[1]+margin] == \
            np.array(config.mask_car_color, dtype=np.int)
        if True in bool_map.flat:
            del matches[i]
            continue
        i += 1


def refine_match_distance(matches, kp1, kp2):
    i = 0
    while i < len(matches):
        m = matches[i]
        end1 = np.array(kp1[m.queryIdx].pt)
        end2 = np.array(kp2[m.trainIdx].pt)
        if np.linalg.norm(end1 - end2) > config.max_norm_dis:
            del matches[i]
            i -= 1
        i += 1

