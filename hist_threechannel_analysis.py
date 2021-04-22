# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import time
import matplotlib.pyplot as plt

plt.ion()

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# 定义对应的命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, default='./standardSample/zhichangai.jpg',
                help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=False, default=2,
                help="width of the left-most object in the image (in inches)")
# ap.add_argument("-e", "--enhance", type=bool, required=False, default=False, help="the key word to open or close
# the enhance function")
ap.add_argument("-e", "--enhance", type=bool, required=False, default=True,
                help="the key word to open or close the enhance function")
args = vars(ap.parse_args())

def plot_histogram(image, title, mask=None):
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0,256])
        plt.plot(hist, color=color)
        plt.xlim([0,256])

# 加载图像并进行灰度处理以及对应的高斯模糊操作
image = cv2.imread(args["image"])
print(image.shape)
image = image[10:, 0:image.shape[1] - 10, :]
plot_histogram(image, "Histogram for Original Image")
plt.show()

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]
cv2.imshow("h", h)
cv2.imshow("s", s)
cv2.imshow("v", v)
cv2.waitKey(0)

(b, g, r) = cv2.split(image)
back = np.zeros(b.shape, dtype=np.uint8)
cv2.imshow("b", cv2.merge([b, back, back]))
cv2.imshow("g", cv2.merge([back,g, back]))
cv2.imshow("r", cv2.merge([back, back, r]))
cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 1)  # 对原始图像中的一些小的边界进行模糊操作

eq = cv2.equalizeHist(gray)
cv2.imshow("Histogram Equalization", np.hstack([gray, eq]))
cv2.waitKey(0)

rever = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
cv2.imshow("rever", rever)
cv2.waitKey(0)

hist = cv2.calcHist([eq], [0], None, [256], [0,256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(hist)
plt.xlim([0,256])
plt.show()

