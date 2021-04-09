import cv2
import numpy as np
import imutils

# 将对应的参考硬币的图使用mask的方法进行完美的提取
img1 = cv2.imread("./biaoding/stand_coin.png")
print(img1.shape)

img1 = imutils.resize(img1, width=100)

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 将原图进行灰度化操作
ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)  # 使用阈值化函数进行mask的制作
cv2.imshow("thresh", thresh)
cv2.waitKey(0)
mask_inv = thresh.copy()
mask = cv2.bitwise_not(thresh)

img2 = cv2.imread("./standardSample/jiazhuangxianrutouzhuangai.jpg")
print(img2.shape)

# img_resize = imutils.resize(img2, width=800)
# print(img_resize.shape)

roi = img2[int(img2.shape[0] / 2):int(img2.shape[0] / 2) + img1.shape[0], 2:2+img1.shape[1]]
img_bg = cv2.bitwise_and(roi, roi, mask = mask)
img_fg = cv2.bitwise_and(img1, img1, mask = mask_inv)
dst = cv2.add(img_fg, img_bg)

img2[int(img2.shape[0] / 2):int(img2.shape[0] / 2) + img1.shape[0], 2:2+img1.shape[1]] = dst

cv2.imshow("dst", dst)
cv2.waitKey(0)

cv2.imwrite("./biaoding/merge.png", img2)
print("save success")

