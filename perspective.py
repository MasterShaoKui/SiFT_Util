import os
import cv2
import json
import exifread
import numpy as np
from dateutil import parser
import scipy.optimize as opt
from time import gmtime, strftime
root_dir = "E:/lane_modeling/the_173"
time_stamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
output_dir = os.path.join(root_dir, "outputs-" + time_stamp)
seq_dir = os.path.join(root_dir, "pics")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
files = os.listdir(seq_dir)
files = [name.split(".")[0] for name in files]
group_name = files[0].split("_")[0]
gps = open(os.path.join(root_dir, "trk.txt")).read().splitlines()
gps = [[float(v) for v in g.split("\t")[:-1]] for g in gps]
gps = np.array(gps)[:, ::-1]
total_pic_len = len(files)  # 图片的总长度
exif_times = []  # 图片的源信息
for i, name in list(enumerate(files)):
    img_name = "{}/{}.jpg".format(seq_dir, name)
    tags = exifread.process_file(open(img_name, 'rb'))
    date = tags["EXIF DateTimeOriginal"].values
    date = date[:10].replace(':', '-')+date[10:]
    date = parser.parse(date)
    subsec = float(tags["EXIF SubSecTime"].values) / 1000000
    exif_times.append(date.timestamp()+subsec)
exif_times = np.array(exif_times)
exif_dtimes = exif_times[1:] - exif_times[:-1]
exif_dtimes = np.clip(exif_dtimes, 0.05, 1e10)
# pl.plot(exif_dtimes)
dtimes = exif_dtimes
dtimes = np.concatenate((dtimes[:1], dtimes))
lines = []
lines_bounds = []
annos = []
for fname in files:
    json_dir = os.path.join(root_dir, "annos")
    anno = json.load(open("{}/{}.jpg.json".format(json_dir, fname), encoding='UTF-8'))
    annos.append(anno)
    ll = len(lines)
    for mark in anno["marks"]:
        line = []
        for l in mark["lines"]:
            line += l
        line = [[l['x'], l['y']] for l in line]
        line = np.array(line)
        if line[0,1] < line[-1,1]:
            line = line[::-1]
        lines.append(line)
    rr = len(lines)
    lines_bounds.append([ll,rr])
lines_bounds_org = np.array(lines_bounds)

img = cv2.imread("{}/{}.jpg".format(seq_dir, files[10]))
lines_bounds = lines_bounds_org.copy()
linese = []
for line in lines:
    linese.append(line[0])
    linese.append(line[int((line.shape[0] + 1) / 2)])
linese = np.array(linese)
linese3 = np.concatenate([linese, np.ones((linese.shape[0], 1))],
                         axis=1)
ex = 0.5
ey = 0.5
newh, neww, neww2 = 1000, 300, 2000
warp_size = (neww2, newh)


def get_pm(ex, ey, cut_rate=0.1, cut_fix=False):
    # end point
    # -----> x
    # | p0        p1
    # |       pe
    # | p6  p4 p5 p7
    # y p3        p2
    # print("ex, ey is :  ", ex, ex)
    dy = 1 - ey
    dx1 = 1 - ex
    dx0 = -ex
    k1, k0 = dx1 / dy, dx0 / dy
    p5x = ex + k1 * dy * cut_rate
    p5y = ey + dy * cut_rate
    p4x = ex + k0 * dy * cut_rate
    p4y = ey + dy * cut_rate
    p0 = [0, 0]
    p1 = [1, 0]
    p5 = [p5x, p5y]
    p4 = [p4x, p4y]
    p6 = [0, p5y]
    p7 = [1, p5y]
    p2 = [1, 1]
    p3 = [0, 1]
    src = np.array([p2, p3, p4, p5], dtype=np.float32)
    dst = np.array([p2, p3, p6, p7], dtype=np.float32)
    if cut_fix:
        dst = np.array([p2, p3, p0, p1], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    scale1 = np.diag([1.0 / img.shape[1], 1.0 / img.shape[0], 1])
    scale2 = np.diag([neww, newh, 1])
    move2 = np.array([[1, 0, neww2 / 2 - neww / 2], [0, 1, 0], [0, 0, 1]])
    nM = np.matmul(move2, np.matmul(scale2, np.matmul(M, scale1)))
    return nM


def calc_loss(ex, ey, lmask=-1):
    nM = get_pm(ex, ey)
    if lmask != -1:
        ml = linese3[lines_bounds[lmask, 0] * 2:lines_bounds[lmask, 1] * 2]
    else:
        ml = linese3
    mlinese3 = np.matmul(nM, ml.T).T
    mlines = mlinese3[:, 0:2] / mlinese3[:, 2:]
    mdlines = mlines[::2] - mlines[1::2]
    mk = mdlines[:, 0] / mdlines[:, 1]
    if lmask != -1:
        mean = np.average(mk)
        for mit in range(10):
            nmk = mk - mean
            nmk = np.argsort(np.abs(nmk))[:max(3, int(mk.shape[0] * 0.5))]
            nmk = mk[nmk]
            nmean = np.average(nmk)
            if nmean == mean:
                break
            mean = nmean
        nmk = nmk - nmean
        loss = np.sum(np.abs(nmk))
        return loss
    mk = np.sort(np.abs(mk))
    loss = np.sum(mk[:int(mk.shape[0] / 2)])
    return loss


def cut_line(line, vcut):
    line = line.copy()
    where = np.where(line[:, 1] < vcut)[0]
    if len(where):
        cut_point = where[0]
        if cut_point == 0:
            line = np.zeros((0, 2))
        else:
            d = line[cut_point - 1] - line[cut_point]
            k = d[0] / d[1]
            t = line[cut_point - 1, 1] - vcut
            line[cut_point] = [-t * k + line[cut_point - 1, 0], vcut]
            line = line[:cut_point + 1]
    return line


result_all = opt.minimize(lambda x: calc_loss(x[0], x[1]), [0.5, 0.5], method="Nelder-Mead", tol=1e-6)
linese = []
zcut = 0.1
for i in range(len(files)):
    bound = lines_bounds[i]
    lb = len(linese)
    for lid in range(bound[0], bound[1]):
        line = lines[lid]
        line = cut_line(line, (result_all['x'][1] + (1 - result_all['x'][1]) * zcut) * img.shape[0])
        if len(line) >= 2:
            linese.append(line[0])
            linese.append(line[int((line.shape[0] + 1) / 2)])
    rb = len(linese)
    lines_bounds[i] = (lb / 2, rb / 2)
linese = np.array(linese)
linese3 = np.concatenate([linese, np.ones((linese.shape[0], 1))], axis=1)


def calc_loss(ex, ey, lmask=-1):
    nM = get_pm(ex, ey)
    if lmask != -1:
        ml = linese3[lines_bounds[lmask, 0] * 2:lines_bounds[lmask, 1] * 2]
    else:
        ml = linese3
    mlinese3 = np.matmul(nM, ml.T).T
    mlines = mlinese3[:, 0:2] / mlinese3[:, 2:]
    mdlines = mlines[::2] - mlines[1::2]
    mk = mdlines[:, 0] / mdlines[:, 1]
    center = warp_size[1] / 2  # 1 or 0
    weights = mlines[::2, 0] - center
    weights = np.max(weights) / np.maximum(weights, 100)
    get_mean = lambda x, y: np.sum(x * y) / np.sum(y)
    if lmask != -1:
        mean = get_mean(mk, weights)  # np.average(mk)
        for mit in range(10):
            nmk = mk - mean
            nmk_id = np.argsort(np.abs(nmk))[:max(3, int(mk.shape[0] * 0.5))]
            nmk = mk[nmk_id]
            nmean = get_mean(nmk, weights[nmk_id])  # np.average(nmk)
            if nmean == mean:
                break
            mean = nmean
        nmk = nmk - nmean
        loss = np.sum(np.abs(nmk))
        return loss
    mk = np.sort(np.abs(mk))
    loss = np.sum(mk[:int(mk.shape[0] / 2)])
    return loss


def warper(x):
    ex = result_all['x'][0]
    ey = x
    loss = 0
    for i in range(len(ey)):
        loss += calc_loss(ex, ey[i], i)
    return loss


opt_num = len(files)
ey = [result_all['x'][1]] * opt_num
ex = result_all['x'][0]

l2_loss = 300

# for it in range(3):
#     # pl.plot(ey)
#     for i in list(range(opt_num)) + list(range(opt_num - 1, 0, -1)):
#         def warper(eyy):
#             loss = calc_loss(ex, eyy, i)
#             if i > 0:
#                 loss += abs(ey[i - 1] - eyy) ** 2 * l2_loss
#             if i < opt_num - 1:
#                 loss += abs(ey[i + 1] - eyy) ** 2 * l2_loss
#             return loss
#         result = opt.minimize(warper, [ey[i]], method="Nelder-Mead", tol=1e-6)
#         ey[i] = result['x'][0]
exyw_dir = output_dir
warp_size = (neww2, newh)
Ms = []  # Perspective matrices
tcut = 0.05
wmls = []
nmls = []
line_types = []


def my_inv(mat_inv, pt):
    pt = np.concatenate((pt, np.ones((pt.shape[0], 1))), axis=1)
    pt = np.matmul(mat_inv, pt.T).T
    return pt[:, :2] / pt[:, 2:]


for i in range(len(files)):
    fname = files[i]
    exx = result_all['x'][0]
    eyy = ey[i]
    nM = get_pm(exx, eyy, tcut, True)
    print(i, exx, eyy, calc_loss(exx, eyy, i))
    img = cv2.imread("{}/{}.jpg".format(seq_dir, files[i]))
    cv2.circle(img, (int(exx * img.shape[1]), int(eyy * img.shape[0])), 5, (0, 0, 255), 3)
    warpd = cv2.warpPerspective(img, nM, warp_size)
    Ms.append(nM)
    # cv2.imwrite("{}/{}.jpg".format(exyw_dir, fname), warpd)


# save perspective matrices
for i, matrix in enumerate(Ms):
    np.savetxt(fname="./t173_pm/" + str(files[i]) + ".txt", X=matrix)

