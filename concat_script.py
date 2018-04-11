import cv2 as cv
import numpy as np
import os
pics_path = "./outputs/"
output_img_name = "concat_img.jpg"
pics_names = os.listdir(pics_path)
if pics_names.count(output_img_name) > 0:
    pics_names.remove(output_img_name)
pic_index = 0
while pic_index < len(pics_names):
    if type(pics_names[pic_index].split("-")[0]) != type(1):
        del pics_names[pic_index]
        pic_index -= 1
    pic_index += 1
pics_names.sort(key=lambda x: int(x.split("_")[0]))
print(pics_names)
img_height = cv.imread(os.path.join(pics_path, pics_names[0]), cv.IMREAD_UNCHANGED).shape[0]
img_width = cv.imread(os.path.join(pics_path, pics_names[0]), cv.IMREAD_UNCHANGED).shape[1]
final_img = np.arange(0).reshape(img_height, 0, 3)
for pic_name in pics_names:
    img = cv.imread(os.path.join(pics_path, pic_name), cv.IMREAD_UNCHANGED)
    final_img = np.hstack((final_img, img))
cv.imwrite(os.path.join(pics_path, output_img_name), final_img)
