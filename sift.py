import cv2 as cv
import numpy as np
from scipy.optimize import minimize
import os
import time
from draw_match import draw_matches_vertical_rgb
import config
from config import root_dir
from config import text_color, text_pos, text_size
from optimize import calculate_perspective_matrix, calculate_linear_t_matrix
from match import refine_match_moving, refine_match_radius, refine_match_op_trend, \
    refine_match_without_car, refine_match_mask_filter, refine_match_distance
import bev
from optical_flow import dense_dual_optical_flow
from mask import mask_roi, mask_no_car


def save_sift_result(pic_name_pre, pic_name_nxt, keys_pre, keys_nxt, matches, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(root_dir, config.o_f_name + "/sift_matching")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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


def choose_bev_point(pic_name_pre, pic_name_nxt, img_pre, img_nxt, matrix, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(root_dir, config.o_f_name + "/chosenps")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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
    dst = np.dot(matrix, np.hstack((dst, np.ones(shape=(dst.shape[0], 1), dtype=np.float32))).T)
    if dst.shape[0] == 3:
        dst = dst / dst[2, :]
        dst = dst[0:2, :]
    dst = dst.T
    for index, p in enumerate(dst):
        cv.circle(img_nxt, (int(p[0]), int(p[1])), 5, (0, 0, 255), 6)
    dst = cv.transform(np.float32([dst]),
                       np.loadtxt(os.path.join(root_dir, "pm/" + pic_name_nxt.split(".")[0] + ".txt"),
                                  dtype=np.float32))
    dst = bev.norm_cv_transform(dst)
    if not config.is_output_chosen_points:
        return points_init, dst
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
    return points_init, dst


def calculate_distance(pre, nxt, mode='v'):
    # pre and nxt are bev points. pre is generated.
    assert pre.shape == nxt.shape, "# bev point is not equal! "

    def model(x, *args):
        x1 = args[0]  # pre
        x2 = args[1]  # nxt
        x1 = x1 + np.array([0, x[0]])
        return np.average(np.linalg.norm(x2-x1, axis=1))
    result = minimize(fun=model, x0=np.array([0]), args=(pre, nxt), method='Nelder-Mead', tol=0.0001)
    print("distance: ", result.x)
    return result.x


def save_optical_flow_result(pic_name_pre, pic_name_nxt, img_pre, img_nxt,
                             ofp_pre, ofp_nxt, mask_pre=None, mask_nxt=None, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(root_dir, config.o_f_name + "/optical_flow")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    color = np.random.randint(0, 255, (ofp_pre.shape[0], 3))
    img_pre = np.copy(img_pre)
    img_nxt = np.copy(img_nxt)
    for i, point in enumerate(ofp_pre):
        img_pre = cv.circle(img_pre, (int(point[0]), int(point[1])), 5, color[i].tolist(), -1)
    mask = np.zeros_like(img_pre)
    for i, (new, old) in enumerate(zip(ofp_nxt, ofp_pre)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        img_nxt = cv.circle(img_nxt, (int(a), int(b)), 5, color[i].tolist(), -1)
    img_nxt = cv.add(img_nxt, mask)
    final_path = os.path.join(output_dir, pic_name_pre.split(".")[0] + "-" + pic_name_nxt.split(".")[0] + ".jpg")
    final_img = np.vstack((img_pre, img_nxt))
    if mask_pre is not None and mask_nxt is not None:
        mask_pn = np.vstack((mask_roi(mask_pre), mask_roi(mask_nxt)))
        final_img = np.hstack((final_img, cv.cvtColor(mask_pn, cv.COLOR_GRAY2BGR)))
    cv.imwrite(final_path, final_img)


def frame_match(pic_name_pre, pic_name_nxt):
    print("start: ", pic_name_pre, " - and - ", pic_name_nxt)
    img_pre = cv.imread(os.path.join(root_dir, "pics/" + pic_name_pre), cv.IMREAD_UNCHANGED)
    img_nxt = cv.imread(os.path.join(root_dir, "pics/" + pic_name_nxt), cv.IMREAD_UNCHANGED)
    mask_pre = cv.imread(os.path.join(root_dir, "mask/" + pic_name_pre), cv.IMREAD_UNCHANGED)
    mask_nxt = cv.imread(os.path.join(root_dir, "mask/" + pic_name_nxt), cv.IMREAD_UNCHANGED)
    # calculate optical flow
    ofp_pre, ofp_nxt, flow = dense_dual_optical_flow(img_pre, img_nxt, mask_pre, mask_nxt)
    if config.is_output_op:
        save_optical_flow_result(pic_name_pre, pic_name_nxt, img_pre, img_nxt, ofp_pre, ofp_nxt, mask_pre, mask_nxt)
    center_avg = (int((img_nxt.shape[1] + img_pre.shape[1]) / 4), int((img_nxt.shape[0] + img_pre.shape[0]) / 4))
    sift = cv.xfeatures2d.SIFT_create()
    keys_pre, des_pre = sift.detectAndCompute(img_pre, mask=mask_no_car(mask_pre))
    keys_nxt, des_nxt = sift.detectAndCompute(img_nxt, mask=mask_no_car(mask_nxt))
    bf = cv.BFMatcher_create(crossCheck=True)
    matches = bf.match(des_pre, des_nxt)
    refine_match_moving(matches, keys_pre, keys_nxt, center_avg)
    refine_match_without_car(matches, keys_pre, keys_nxt)
    refine_match_mask_filter(matches, keys_pre, keys_nxt, mask_pre, mask_nxt)
    refine_match_distance(matches, keys_pre, keys_nxt)
    refine_match_op_trend(matches, keys_pre, keys_nxt, flow)
    refine_match_radius(matches, keys_pre, keys_nxt)
    matches = sorted(matches, key=lambda x: x.distance)
    if config.is_output_sift_matching:
        save_sift_result(pic_name_pre, pic_name_nxt, keys_pre, keys_nxt, matches)
    x1 = np.zeros(shape=(0, 2))
    x2 = np.zeros(shape=(0, 2))
    for m in matches:
        pt1 = np.array(keys_pre[m.queryIdx].pt)
        pt2 = np.array(keys_nxt[m.trainIdx].pt)
        x1 = np.vstack((x1, pt1.reshape(1, 2)))
        x2 = np.vstack((x2, pt2.reshape(1, 2)))
    dis_list = list()
    matrix_list = list()
    if x1.shape[0] >= 4 and x2.shape[0] >= 4:
        matrix = calculate_perspective_matrix(x1, x2)
        bev_p_pre, bev_p_nxt = choose_bev_point(pic_name_pre, pic_name_nxt, img_pre, img_nxt, matrix)
        dis_list.append(calculate_distance(bev_p_pre, bev_p_nxt))
        matrix_list.append(matrix)
    # an alternative way to generate distance using op points.
    x1 = np.vstack((ofp_pre, x1))
    x2 = np.vstack((ofp_nxt, x2))
    if x1.shape[0] >= 4 and x2.shape[0] >= 4:
        matrix_with_op = calculate_perspective_matrix(x1, x2)
        bev_p_pre, bev_p_nxt = choose_bev_point(pic_name_pre, pic_name_nxt, img_pre, img_nxt, matrix_with_op,
                                                output_dir=os.path.join(root_dir, config.o_f_name + "/chosenps_op"))
        dis_list.append(calculate_distance(bev_p_pre, bev_p_nxt))
        matrix_list.append(matrix_with_op)
    dis_index = 0
    while dis_index < len(dis_list):
        if dis_list[dis_index] <= 0:
            del dis_list[dis_index]
            del matrix_list[dis_index]
            dis_index -= 1
        dis_index += 1
    if len(dis_list) == 0:
        print("error! no valid distance! ")
        return None
    # pixel is discrete, so the distance should be an integer
    assert len(dis_list) == len(matrix_list), "len(dis_list) == len(matrix_list)"
    dis = int(np.average(np.array(dis_list, dtype=np.float32)))
    final_matrix = np.zeros_like(matrix_list[0], dtype=np.float32)
    for m in matrix_list:
        final_matrix += m
    final_matrix = final_matrix / len(matrix_list)
    if config.is_output_frame_match:
        save_frame_match(pic_name_pre, pic_name_nxt, dis)
    return final_matrix


def save_frame_match(pic_name_pre, pic_name_nxt, dis, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(root_dir, config.o_f_name + "/frame_match")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    bev_img_pre = cv.imread(os.path.join(root_dir, "bev_pics/" + pic_name_pre), cv.IMREAD_UNCHANGED)
    bev_img_nxt = cv.imread(os.path.join(root_dir, "bev_pics/" + pic_name_nxt), cv.IMREAD_UNCHANGED)
    final_dir = os.path.join(output_dir, pic_name_pre.split(".")[0].split("_")[-1]
                             + "-" + pic_name_nxt.split(".")[0].split("_")[-1] + ".jpg")
    if dis > bev_img_pre.shape[0]:
        final_img = np.vstack((bev_img_nxt, bev_img_pre))
    else:
        margin_shape = (dis, bev_img_pre.shape[1], bev_img_pre.shape[2])
        margin = np.zeros(shape=margin_shape)
        bev_img_pre = np.vstack((margin, bev_img_pre))
        bev_img_nxt = np.vstack((bev_img_nxt, margin))
        final_img = (bev_img_pre + bev_img_nxt) / 2
    cv.imwrite(final_dir, final_img)


def save_core_matrix(matrix, index, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(root_dir, config.o_f_name + "/core_matrices")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(output_dir + "/" + str(index) + ".txt", matrix, fmt="%.6f")


if __name__ == '__main__':
    config.is_output_sift_matching = False
    config.is_output_chosen_points = True
    config.is_output_op = False
    config.is_output_frame_match = True
    files = os.listdir(root_dir + "pics/")
    config.o_f_name = "5-4-final"
    if not os.path.exists(os.path.join(root_dir, config.o_f_name)):
        os.makedirs(os.path.join(root_dir, config.o_f_name))
    i = 0
    while i < len(files)-1:
        all_matrices = open(os.path.join(root_dir, config.o_f_name) + "/matrices.txt", mode='a')
        time_start = time.time()
        m = frame_match(files[i], files[i + 1])
        print(m)
        if m is None:
            m = np.zeros(shape=(2, 3))
        save_core_matrix(m, i)
        print(np.array2string(m, formatter={'float_kind': lambda x: "%.3f" % x}))
        all_matrices.write(np.array2string(m, formatter={'float_kind': lambda x: "%.3f" % x}))
        all_matrices.write("\r\n")
        time_end = time.time()
        print("总运行时间： " + str(time_end - time_start) + "s")
        i += 1
        all_matrices.close()
