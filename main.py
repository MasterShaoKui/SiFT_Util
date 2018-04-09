import cv2 as cv
import os
from draw_match import draw_matches_vertical_rgb
outputs_dir = "./outputs/"
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
text_pos = 100
text_size = 1
text_color = (255, 0, 0)
img_65 = cv.imread("./data/65.jpg", cv.IMREAD_UNCHANGED)
img_66 = cv.imread("./data/66.jpg", cv.IMREAD_UNCHANGED)
sift = cv.xfeatures2d.SIFT_create()
keys_img_65, des_img_65 = sift.detectAndCompute(img_65, None)
keys_img_66, des_img_66 = sift.detectAndCompute(img_66, None)
print(len(keys_img_65))
print(des_img_65.shape)
bf = cv.BFMatcher_create()
matches = bf.match(des_img_65, des_img_66)
matches = sorted(matches, key=lambda x: x.distance)
print(matches[0])
print(type(matches[0]))
for i in (1, 5, 10, 20, 30, 100):
    match_img = draw_matches_vertical_rgb(img_65, keys_img_65, img_66, keys_img_66, matches[: i])
    cv.putText(match_img, "pre", (text_pos, text_pos),
               cv.FONT_HERSHEY_COMPLEX, text_size, (0, 255, 0), 2, cv.LINE_4)
    cv.putText(match_img, "next", (text_pos, match_img.shape[0] - text_pos),
               cv.FONT_HERSHEY_COMPLEX, text_size, (0, 255, 0), 2, cv.LINE_4)
    cv.putText(match_img, "matched_num_" + str(i), (text_pos, text_pos*2),
               cv.FONT_HERSHEY_COMPLEX, text_size, text_color, 2, cv.LINE_4)
    cv.imwrite("./outputs/" + str(i) + "_map.jpg", match_img)
match_img = draw_matches_vertical_rgb(img_65, keys_img_65, img_66, keys_img_66, matches)
cv.imwrite("./outputs/" + str(len(matches)) + "_map_full.jpg", match_img)
# cv.namedWindow("match_img", cv.WINDOW_NORMAL)
# cv.imshow("match_img", match_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# key points 是 list，里面都是Keypoint对象
# descriptor 是ndarray
# 关键点与关键点的描述是对应的,keys的每一个元素对应descriptor的每一行
# for match in matches:
#     print(match.queryIdx, match.trainIdx, match.imgIdx, match.distance)
# 左边随机取一个点，右边找最近邻，再找第二近邻
