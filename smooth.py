import numpy as np
import cv2 as cv
import time
import os
from scipy.ndimage.filters import gaussian_filter1d
import config
from sift import choose_bev_point, calculate_distance, save_frame_match


def smooth_frame_matrices_gaussian(matrices):
    assert len(matrices.shape) == 3, "len(matrices.shape) should be 3! "
    zeros = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
    for i in range(matrices.shape[0]):
        if np.array_equal(matrices[i], zeros):
            print("found zero matrix")
            j = i+1
            while j < matrices.shape[0] and np.array_equal(matrices[j], zeros):
                j += 1
            if j < matrices.shape[0] and i > 0:
                matrices[i] = matrices[j] + matrices[i-1]
                matrices[i] = matrices[i] / 2
    for i in range(matrices.shape[1]):
        for j in range(matrices.shape[2]):
            matrices[:, i, j] = gaussian_filter1d(matrices[:, i, j], 0.5)
    return matrices


def extract_matrices(folder):
    matrices = np.zeros(shape=(0, 2, 3))
    files = os.listdir(folder)
    files.sort(key=lambda x: int(x.split(".")[0]))
    for f_name in files:
        matrix = np.loadtxt(folder + "/" + f_name, dtype=np.float32).reshape(1, 2, 3)
        matrices = np.vstack((matrices, matrix))
    return matrices


def frame_match(pic_name_pre, pic_name_nxt, matrix=None, dis=None):
    assert (matrix is not None and dis is None) or (matrix is None and dis is not None), \
        "matrix and dis should have one! "
    print("start: ", pic_name_pre, " - and - ", pic_name_nxt)
    img_pre = cv.imread(os.path.join(config.root_dir, "pics/" + pic_name_pre), cv.IMREAD_UNCHANGED)
    img_nxt = cv.imread(os.path.join(config.root_dir, "pics/" + pic_name_nxt), cv.IMREAD_UNCHANGED)
    if dis is None:
        bev_p_pre, bev_p_nxt = choose_bev_point(pic_name_pre, pic_name_nxt, img_pre, img_nxt, matrix)
        dis = calculate_distance(bev_p_pre, bev_p_nxt)
        if dis < 0:
            dis = 0
    dis = int(dis)
    if config.is_output_frame_match:
        save_frame_match(pic_name_pre, pic_name_nxt, dis)
    print("distance: ", dis)


def frame_distance(pic_name_pre, pic_name_nxt, matrix, pre_dis=None):
    img_pre = cv.imread(os.path.join(config.root_dir, "pics/" + pic_name_pre), cv.IMREAD_UNCHANGED)
    img_nxt = cv.imread(os.path.join(config.root_dir, "pics/" + pic_name_nxt), cv.IMREAD_UNCHANGED)
    bev_p_pre, bev_p_nxt = choose_bev_point(pic_name_pre, pic_name_nxt, img_pre, img_nxt, matrix)
    dis = calculate_distance(bev_p_pre, bev_p_nxt)
    if dis <= config.minimal_bev_dis:
        dis = 0.
        if pre_dis is not None:
            dis = pre_dis
    dis = int(dis)
    return dis


config.is_output_chosen_points = False
config.is_output_frame_match = True
files = os.listdir(config.root_dir + "pics/")
config.o_f_name = "5-4-final_nxt"
if not os.path.exists(os.path.join(config.root_dir, config.o_f_name)):
    os.makedirs(os.path.join(config.root_dir, config.o_f_name))
matrices = extract_matrices("E:/lane_modeling/the_173/5-4-final/core_matrices")
matrices = smooth_frame_matrices_gaussian(matrices)
distances = list()
i = 0
while i < len(files)-1:
    all_matrices = open(os.path.join(config.root_dir, config.o_f_name) + "/matrices.txt", mode='a')
    if i == 0:
        distances.append(frame_distance(files[i], files[i + 1], matrices[i]))
    else:
        distances.append(frame_distance(files[i], files[i + 1], matrices[i], distances[i-1]))
    i += 1
i = 0
distances = np.array(distances, np.float32)
distances = gaussian_filter1d(distances, 0.5)
np.savetxt("dis.txt", distances, fmt="%.6f")
while i < len(files)-1:
    all_matrices = open(os.path.join(config.root_dir, config.o_f_name) + "/matrices.txt", mode='a')
    time_start = time.time()
    frame_match(files[i], files[i + 1], dis=distances[i])
    time_end = time.time()
    print("总运行时间： " + str(time_end - time_start) + "s")
    i += 1
