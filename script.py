import cv2 as cv
a = cv.imread("./data/65.jpg", cv.IMREAD_UNCHANGED)
b = cv.imread("./data/66.jpg", cv.IMREAD_UNCHANGED)
cv.imwrite("./data/65-66.jpg", a*0.5+b*0.5)
