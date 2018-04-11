import numpy as np
import cv2 as cv

newh, neww, neww2 = 1000, 300, 1000
warp_size = (neww2, newh)


def get_pm(ex, ey, img, cut_rate=0.1, cut_fix=False):
    # end point
    # -----> x
    # | p0        p1
    # |       pe
    # | p6  p4 p5 p7
    # y p3        p2
    dy = 1-ey
    dx1 = 1-ex
    dx0 = -ex
    k1, k0 = dx1 / dy, dx0 / dy
    p5x = ex + k1*dy*cut_rate
    p5y = ey + dy*cut_rate
    p4x = ex + k0*dy*cut_rate
    p4y = ey + dy*cut_rate
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
    matrix = cv.getPerspectiveTransform(src, dst)
    scale1 = np.diag([1.0/img.shape[1], 1.0/img.shape[0], 1])
    scale2 = np.diag([neww, newh, 1])
    move2 = np.array([[1, 0, neww2/2-neww/2], [0, 1, 0], [0, 0, 1]])
    matrix = np.matmul(move2, np.matmul(scale2, np.matmul(matrix, scale1)))
    return matrix


tcut = 0.1
Ms = []
for i in range(len(files)):
    fname = files[i]
    exx = result_all['x'][0]
    eyy = ey[i]
    # eyy = 0.6117
    img = cv.imread("{}/{}.jpg".format(seq_dir, files[i]))
    nM = get_pm(exx, eyy, img, tcut, True)
    cv.circle(img, (int(exx * img.shape[1]), int(eyy * img.shape[0])), 5, (0, 0, 255), 3)
    warpd = cv.warpPerspective(img, nM, warp_size)
    Ms.append(nM)
    vcut = ((1 - eyy) * tcut + eyy) * img.shape[0]
    cv.imwrite("{}/{}.jpg".format(exy_dir, fname), img)
    cv.imwrite("{}/{}.jpg".format(exyw_dir, fname), warpd)
