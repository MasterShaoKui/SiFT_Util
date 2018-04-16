import cv2 as cv
import numpy as np
import os
from time import gmtime, strftime
from draw_match import draw_matches_vertical_rgb
import config
from config import root_dir
from config import text_color, text_pos, text_size, is_output_img
from optimize import calculate_perspective_matrix
from match import refine_match_moving, refine_match_radius, \
    refine_match_without_car, refine_match_mask_filter, refine_match_distance
import bev


def save_sift_result(pic_name_pre, pic_name_nxt, keys_pre, keys_nxt, matches,
                     output_dir=os.path.join(root_dir, "outputs")):
    img_pre = cv.imread(os.path.join(root_dir, "pics/" + pic_name_pre), cv.IMREAD_UNCHANGED)
    img_nxt = cv.imread(os.path.join(root_dir, "pics/" + pic_name_nxt), cv.IMREAD_UNCHANGED)
    concat_img_name = pic_name_pre.split(".")[0] + "-" + pic_name_nxt.split(".")[0] + ".jpg"
    match_img_full = draw_matches_vertical_rgb(img_pre, keys_pre, img_nxt, keys_nxt, matches)
    final_img = np.arange(0).reshape(match_img_full.shape[0], 0, 3)
    for i in (1, 5, 10, 15, 20, 30, 50, 100):
        if i > len(matches):
            break
        if not config.is_output_img:
            break
        match_img = draw_matches_vertical_rgb(img_pre, keys_pre, img_nxt, keys_nxt, matches[: i])
        cv.putText(match_img, "pre", (text_pos, text_pos),
                   cv.FONT_HERSHEY_COMPLEX, text_size, (0, 255, 0), 2, cv.LINE_4)
        cv.putText(match_img, "next", (text_pos, match_img.shape[0] - text_pos),
                   cv.FONT_HERSHEY_COMPLEX, text_size, (0, 255, 0), 2, cv.LINE_4)
        cv.putText(match_img, "matched_num_" + str(i), (text_pos, text_pos * 2),
                   cv.FONT_HERSHEY_COMPLEX, text_size, text_color, 2, cv.LINE_4)
        final_img = np.hstack((final_img, match_img))
    if config.is_output_img:
        final_img = np.hstack((final_img, match_img_full))
    cv.imwrite(os.path.join(output_dir, concat_img_name), final_img)


def save_chosen_point_result(pic_name_pre, pic_name_nxt, img_pre, img_nxt, matrix,
                             output_dir=os.path.join(root_dir, "outputs")):
    # 开始计算用于帧间匹配的点
    bev_img_pre = cv.imread(os.path.join(root_dir, "bev_pics/" + pic_name_pre), cv.IMREAD_UNCHANGED)
    bev_img_nxt = cv.imread(os.path.join(root_dir, "bev_pics/" + pic_name_nxt), cv.IMREAD_UNCHANGED)
    points_init = bev.get_bev_base_points(bev_img_pre)
    for index, p in enumerate(points_init):
        cv.putText(bev_img_pre, str(index), (int(p[0]), int(p[1])),
                   cv.FONT_HERSHEY_COMPLEX, text_size, (0, 255, 0), 2, cv.LINE_4)
    dst = cv.transform(np.float32([points_init]),
                       np.linalg.inv(np.loadtxt(os.path.join(root_dir, "pm/" + pic_name_pre.split(".")[0] + ".txt"),
                                                dtype=np.float32)))
    dst = bev.norm_cv_transform(dst)
    for index, p in enumerate(dst):
        cv.circle(img_pre, (int(p[0]), int(p[1])), 5, (0, 0, 255), 6)
    dst = cv.transform(np.float32([dst]), matrix)
    dst = bev.norm_cv_transform(dst)
    for index, p in enumerate(dst):
        cv.circle(img_nxt, (int(p[0]), int(p[1])), 5, (0, 0, 255), 6)
    dst = cv.transform(np.float32([dst]),
                       np.loadtxt(os.path.join(root_dir, "pm/" + pic_name_nxt.split(".")[0] + ".txt"),
                                  dtype=np.float32))
    dst = bev.norm_cv_transform(dst)
    for index, p in enumerate(dst):
        cv.putText(bev_img_nxt, str(index), (int(p[0]), int(p[1])),
                   cv.FONT_HERSHEY_COMPLEX, text_size, (0, 255, 0), 2, cv.LINE_4)
    assert bev_img_pre.shape == bev_img_nxt.shape, "Bev should have the same size! "
    img_pre_resized = cv.resize(img_pre, dsize=(bev_img_pre.shape[1], bev_img_pre.shape[0]))
    img_nxt_resized = cv.resize(img_nxt, dsize=(bev_img_pre.shape[1], bev_img_pre.shape[0]))
    final_img = np.hstack((np.vstack((img_pre_resized, img_nxt_resized)), np.vstack((bev_img_pre, bev_img_nxt))))
    final_img_name = pic_name_pre.split(".")[0].split("_")[-1] + "-" + \
                     pic_name_nxt.split(".")[0].split("_")[-1] + ".jpg"
    final_img_dir = os.path.join(output_dir, final_img_name)
    cv.imwrite(final_img_dir, final_img)


def frame_match(pic_name_pre, pic_name_nxt, output_dir=os.path.join(root_dir, "outputs")):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("start: ", pic_name_pre, " - and - ", pic_name_nxt)
    img_pre = cv.imread(os.path.join(root_dir, "pics/" + pic_name_pre), cv.IMREAD_UNCHANGED)
    img_nxt = cv.imread(os.path.join(root_dir, "pics/" + pic_name_nxt), cv.IMREAD_UNCHANGED)
    center_avg = (int((img_nxt.shape[1] + img_pre.shape[1]) / 4), int((img_nxt.shape[0] + img_pre.shape[0]) / 4))
    sift = cv.xfeatures2d.SIFT_create()
    keys_pre, des_pre = sift.detectAndCompute(img_pre, None)
    keys_nxt, des_nxt = sift.detectAndCompute(img_nxt, None)
    bf = cv.BFMatcher_create(crossCheck=True)
    matches = bf.match(des_pre, des_nxt)
    refine_match_moving(matches, keys_pre, keys_nxt, center_avg)
    refine_match_without_car(matches, keys_pre, keys_nxt)
    refine_match_mask_filter(matches, keys_pre, keys_nxt,
                             os.path.join(root_dir, "mask/" + pic_name_pre),
                             os.path.join(root_dir, "mask/" + pic_name_pre))
    refine_match_distance(matches, keys_pre, keys_nxt)
    refine_match_radius(matches, keys_pre, keys_nxt)
    matches = sorted(matches, key=lambda x: x.distance)
    x1 = np.zeros(shape=(0, 2))
    x2 = np.zeros(shape=(0, 2))
    for m in matches:
        pt1 = np.array(keys_pre[m.queryIdx].pt)
        pt2 = np.array(keys_nxt[m.trainIdx].pt)
        x1 = np.vstack((x1, pt1.reshape(1, 2)))
        x2 = np.vstack((x2, pt2.reshape(1, 2)))
    matrix = calculate_perspective_matrix(x1, x2)
    if config.is_output_sift_matching:
        save_sift_result(pic_name_pre, pic_name_nxt, keys_pre, keys_nxt, matches,
                         output_dir=os.path.join(root_dir, "sift_matching"))
    if config.is_output_chosen_points:
        save_chosen_point_result(pic_name_pre, pic_name_nxt, img_pre, img_nxt, matrix,
                                 output_dir=os.path.join(root_dir, "chosenps"))


config.is_output_sift_matching = True
config.is_output_chosen_points = True
files = os.listdir(root_dir + "pics/")
for i in range(len(files) - 1):
    frame_match(files[i], files[i+1])

