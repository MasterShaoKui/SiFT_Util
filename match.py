import numpy as np
import config
from config import mask_margin as margin


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
        if end1[1] > config.match_front_car:
            del matches[i]
            i -= 1
        i += 1
    while i < len(matches):
        m = matches[i]
        end2 = np.array(kp2[m.trainIdx].pt)
        if end2[1] > config.match_front_car:
            del matches[i]
            i -= 1
        i += 1


def refine_match_mask_filter(matches, kp1, kp2, mask1, mask2):
    i = 0
    while i < len(matches):
        m = matches[i]
        end1 = np.array(kp1[m.queryIdx].pt, dtype=np.int)
        # 不要过于边缘的点
        if end1[0] - margin <= 0 or end1[0] + margin >= mask1.shape[0] \
                or end1[1] - margin <= 0 or end1[1] + margin >= mask1.shape[1]:
            del matches[i]
            continue
        if np.array(config.mask_car_color, dtype=np.int) in \
                mask1[end1[0]-margin:end1[0]+margin, end1[1]-margin:end1[1]+margin]:
            del matches[i]
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
        if np.array(config.mask_car_color, dtype=np.int) in \
            mask2[end2[0]-margin:end2[0]+margin, end2[1]-margin:end2[1]+margin]:
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


def refine_match_radius(matches, kp1, kp2):
    i = 0
    while i < len(matches):
        m = matches[i]
        end1 = np.array(kp1[m.queryIdx].pt)
        end2 = np.array(kp2[m.trainIdx].pt)
        for j in range(i):
            m_cp = matches[j]
            end1_cp = np.array(kp1[m_cp.queryIdx].pt)
            end2_cp = np.array(kp2[m_cp.trainIdx].pt)
            if np.linalg.norm(end1 - end1_cp) < config.max_grid or np.linalg.norm(end2 - end2_cp) < config.max_grid:
                del matches[i]
                i -= 1
                break
        i += 1


def refine_match_op_trend(matches, kp1, kp2, flow):
    i = 0
    while i < len(matches):
        m = matches[i]
        end1 = np.array(kp1[m.queryIdx].pt)
        end2 = np.array(kp2[m.trainIdx].pt)
        end = end2 - end1
        # p2 = p1 + flow[int(p[1]), int(p[0])]
        # p2 = p1 + flow[int(p[1]), int(p[0])]
        vec = flow[int(end1[0]), int(end1[1])]
        # to unit
        end = end / np.linalg.norm(end)
        vec = vec / np.linalg.norm(vec)
        if np.dot(end, vec) < config.mask_op_trend_thresh:
            del matches[i]
            i -= 1
        i += 1
