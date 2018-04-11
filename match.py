import numpy as np


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
