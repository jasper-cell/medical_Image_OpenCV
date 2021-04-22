import cv2 as cv
import numpy as np

# img = np.uint8(img*255)
# src = img
src = cv.imread("./standardSample/zigongjiliuFuQiang.jpg")
src = cv.resize(src, (0, 0), fx=0.5, fy=0.5)
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
img = src
r = cv.selectROI('input', img, False)  # 返回 (x_min, y_min, w, h)

# roi区域
roi = src[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

# 原图mask
maskc = np.zeros(src.shape[:2], dtype=np.uint8)

# 矩形roi
rect = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))  # 包括前景的矩形，格式为(x,y,w,h)

bgdmodel = np.zeros((1, 65), np.float64)  # bg模型的临时数组
fgdmodel = np.zeros((1, 65), np.float64)  # fg模型的临时数组
imgc = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
cv.grabCut(imgc, maskc, rect, bgdmodel, fgdmodel, 11, mode=cv.GC_INIT_WITH_RECT)

# 提取前景和可能的前景区域
mask2 = np.where((maskc == 1) + (maskc == 3), 255, 0).astype('uint8')

print(mask2.shape)
cv.imshow("mask", mask2)

result = cv.bitwise_and(src, src, mask=mask2)
cv.imwrite('result.jpg', result)
cv.imwrite('roi.jpg', roi)

cv.imshow('roi', roi)
cv.imshow("result", result)
cv.waitKey(0)
cv.destroyAllWindows()
label = np.uint8(result>0)