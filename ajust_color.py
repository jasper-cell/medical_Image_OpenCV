import argparse
import imutils
import cv2
import numpy as np

# 定义对应的命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, default='./standardSample/shiguanai.jpg',
    help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=False, default=2,
    help="width of the left-most object in the image (in inches)")
ap.add_argument("-e", "--enhance", type=bool, required=False, default=True, help="the key word to open or close the enhance function")
args = vars(ap.parse_args())


sample_A = (138, 681)
sample_B = (179, 682)

# 加载图像并进行灰度处理以及对应的高斯模糊操作
image = cv2.imread(args["image"])
print(image.shape)
image = image[10:, 0:image.shape[1]-10, :]
print(image.shape)

# 定义对应的信号函数
def hsv(arg):
    # 定义hsv空间的相应值
    LH = cv2.getTrackbarPos("LH","TrackBars")
    LS = cv2.getTrackbarPos("LS","TrackBars")
    LV = cv2.getTrackbarPos("LV","TrackBars")
    LH = cv2.getTrackbarPos("LH","TrackBars")
    UH = cv2.getTrackbarPos("UH","TrackBars")
    US = cv2.getTrackbarPos("US","TrackBars")
    UV = cv2.getTrackbarPos("UV","TrackBars")

    # 定义对比度和亮度的调节值
    Ncnum = cv2.getTrackbarPos("NContrast", "TrackBars")
    Nbnum = cv2.getTrackbarPos("NBrightness", "TrackBars")
    Pcnum = cv2.getTrackbarPos("PContrast", "TrackBars")
    Pbnum = cv2.getTrackbarPos("PBrightness", "TrackBars")

    # 将hsv空间的底层值和顶层值进行归纳总结
    lower = np.array([LH, LS, LV])
    upper = np.array([UH, US, UV])

    # 使用对应的参数对图像进行操作，从而取得相应的roi区域
    adjusted = imutils.adjust_brightness_contrast(image, contrast=float(Ncnum), brightness=float(-Nbnum))
    adjusted2 = imutils.adjust_brightness_contrast(adjusted, contrast=float(Pcnum), brightness=float(Pbnum))
    image_hsv = cv2.cvtColor(adjusted2, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower, upper)
    image_res = cv2.bitwise_and(adjusted2, adjusted2, mask=mask)
    cv2.imshow("res", adjusted2)
    cv2.imshow("hsv", image_hsv)
    cv2.imshow("return", image_res)

# 创建对应的自定义的对比度和亮度的调节器
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 480)
cv2.createTrackbar("LH","TrackBars",0,255,hsv)
cv2.createTrackbar("LS","TrackBars",0,255,hsv)
cv2.createTrackbar("LV","TrackBars",0,255,hsv)
cv2.createTrackbar("UH","TrackBars",77,255,hsv)
cv2.createTrackbar("US","TrackBars",255,255,hsv)
cv2.createTrackbar("UV","TrackBars",255,255,hsv)
cv2.createTrackbar("NBrightness", "TrackBars", 0, 256, hsv)
cv2.createTrackbar("NContrast", "TrackBars", 0, 256, hsv)
cv2.createTrackbar("PBrightness", "TrackBars", 0, 256, hsv)
cv2.createTrackbar("PContrast", "TrackBars", 0, 256, hsv)
cv2.waitKey(0)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
