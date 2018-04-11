import cv2 as cv
import numpy as np
import os
from time import gmtime, strftime
from draw_match import draw_matches_vertical_rgb
from config import text_color, text_pos, text_size
from optimize import calculate_perspective_matrix
from match import refine_match_moving, refine_match_without_car
time_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
outputs_dir = os.path.join("./outputs/", time_stamp)
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
img_pre = cv.imread("./data/65.jpg", cv.IMREAD_UNCHANGED)
img_nxt = cv.imread("./data/66.jpg", cv.IMREAD_UNCHANGED)
# calculate the center point
center_pre = (int(img_pre.shape[1]/2), int(img_pre.shape[0]/2))
center_nxt = (int(img_nxt.shape[1]/2), int(img_nxt.shape[0]/2))
center_avg = (int((img_nxt.shape[1] + img_pre.shape[1])/4), int((img_nxt.shape[0] + img_pre.shape[0])/4))
print(center_avg)
sift = cv.xfeatures2d.SIFT_create()
keys_img_65, des_img_65 = sift.detectAndCompute(img_pre, None)
keys_img_66, des_img_66 = sift.detectAndCompute(img_nxt, None)
'''
Second param is boolean variable, crossCheck which is False by default. 
If it is true, 
Matcher returns only those matches with value (i,j) 
such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa. 
That is, the two features in both sets should match each other. 
It provides consistant result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.
'''
bf = cv.BFMatcher_create(crossCheck=True)
matches = bf.match(des_img_65, des_img_66)
refine_match_moving(matches, keys_img_65, keys_img_66, center_avg)
refine_match_without_car(matches, keys_img_65, keys_img_66)
matches = sorted(matches, key=lambda x: x.distance)
x1 = np.zeros(shape=(0, 2))
x2 = np.zeros(shape=(0, 2))
for m in matches[0:4]:
    pt1 = np.array(keys_img_65[m.queryIdx].pt)
    pt2 = np.array(keys_img_66[m.trainIdx].pt)
    x1 = np.vstack((x1, pt1.reshape(1, 2)))
    x2 = np.vstack((x2, pt2.reshape(1, 2)))
calculate_perspective_matrix(x1, x2)
p_img = cv.warpPerspective(img_pre, calculate_perspective_matrix(x1, x2),
                           (img_pre.shape[1], img_pre.shape[0]))
cv.imwrite(os.path.join(outputs_dir, "perspective.jpg"), p_img)
cv.imwrite(os.path.join(outputs_dir, "img_nxt.jpg"), img_nxt)
cv.imwrite(os.path.join(outputs_dir, "overlap.jpg"), img_nxt*0.5+p_img*0.5)
# end1 = tuple(kp1[m.queryIdx].pt)
# end2 = tuple(kp2[m.trainIdx].pt)
for i in (1, 4, 5, 8, 10, 20):
    match_img = draw_matches_vertical_rgb(img_pre, keys_img_65, img_nxt, keys_img_66, matches[: i])
    cv.putText(match_img, "pre", (text_pos, text_pos),
               cv.FONT_HERSHEY_COMPLEX, text_size, (0, 255, 0), 2, cv.LINE_4)
    cv.putText(match_img, "next", (text_pos, match_img.shape[0] - text_pos),
               cv.FONT_HERSHEY_COMPLEX, text_size, (0, 255, 0), 2, cv.LINE_4)
    cv.putText(match_img, "matched_num_" + str(i), (text_pos, text_pos*2),
               cv.FONT_HERSHEY_COMPLEX, text_size, text_color, 2, cv.LINE_4)
    cv.imwrite(os.path.join(outputs_dir, str(i) + "_map.jpg"), match_img)
match_img = draw_matches_vertical_rgb(img_pre, keys_img_65, img_nxt, keys_img_66, matches)
cv.imwrite(os.path.join(outputs_dir, str(len(matches)) + "_map_full.jpg"), match_img)
'''
key points 是 list，里面都是Keypoint对象
descriptor 是ndarray
关键点与关键点的描述是对应的,keys的每一个元素对应descriptor的每一行
for match in matches:
    print(match.queryIdx, match.trainIdx, match.imgIdx, match.distance)
左边随机取一个点，右边找最近邻，再找第二近邻
akey = keys_img_65[0]
print(type(akey.angle), type(akey.class_id),
      type(akey.octave), type(akey.pt),
      type(akey.response), type(akey.size))
print(akey.angle, akey.class_id, akey.octave, akey.pt, akey.response, akey.size)

bfmatcher 是不对称的双向匹配，现在需要对称的双向匹配。
'''