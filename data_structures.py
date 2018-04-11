import cv2 as cv
# DMatch
# Class for matching keypoint descriptors.
# query descriptor index,
# train descriptor index,
# train image index,
# and distance between descriptors.

# match 使用了欧氏距离
# print(matches[0])
# print(type(matches[0]))
# m = matches[0]
# print(m.distance, m.queryIdx, m.trainIdx)
# dist = np.linalg.norm(des_img_65[1862] - des_img_66[1909])
# print("dist: ", dist)
cv.DMatch()

# KeyPoint
# print(type(akey.angle), type(akey.class_id),
#       type(akey.octave), type(akey.pt),
#       type(akey.response), type(akey.size))
# print(akey.angle, akey.class_id, akey.octave, akey.pt, akey.response, akey.size)
cv.KeyPoint()
