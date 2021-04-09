# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import time

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# 定义对应的命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False, default='./standardSample/jiazhuangxianrutouzhuangai.jpg',
                help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=False, default=2,
                help="width of the left-most object in the image (in inches)")
# ap.add_argument("-e", "--enhance", type=bool, required=False, default=False, help="the key word to open or close
# the enhance function")
ap.add_argument("-e", "--enhance", type=bool, required=False, default=True,
                help="the key word to open or close the enhance function")
args = vars(ap.parse_args())

sample_A = (138, 681)
sample_B = (179, 682)

# 加载图像并进行灰度处理以及对应的高斯模糊操作
image = cv2.imread(args["image"])
print(image.shape)
image = image[10:, 0:image.shape[1] - 10, :]

# 定义对应的信号函数
def hsv(arg):
    global image_res
    # 定义hsv空间的相应值
    LH = cv2.getTrackbarPos("LH", "TrackBars")
    LS = cv2.getTrackbarPos("LS", "TrackBars")
    LV = cv2.getTrackbarPos("LV", "TrackBars")
    LH = cv2.getTrackbarPos("LH", "TrackBars")
    UH = cv2.getTrackbarPos("UH", "TrackBars")
    US = cv2.getTrackbarPos("US", "TrackBars")
    UV = cv2.getTrackbarPos("UV", "TrackBars")

    # 定义对比度和亮度的调节值
    Ncnum = cv2.getTrackbarPos("NContrast", "TrackBars")
    Nbnum = cv2.getTrackbarPos("NBrightness", "TrackBars")
    Pcnum = cv2.getTrackbarPos("PContrast", "TrackBars")
    Pbnum = cv2.getTrackbarPos("PBrightness", "TrackBars")

    # 将hsv空间的底层值和顶层值进行归纳总结
    lower = np.array([LH, LS, LV])
    upper = np.array([UH, US, UV])

    # 使用对应的参数对图像进行操作，从而取得相应的roi区域
    image_blur = cv2.GaussianBlur(image, (5,5), 0)
    adjusted = imutils.adjust_brightness_contrast(image_blur, contrast=float(Ncnum), brightness=float(-Nbnum))
    adjusted2 = imutils.adjust_brightness_contrast(adjusted, contrast=float(Pcnum), brightness=float(Pbnum))
    image_hsv = cv2.cvtColor(adjusted2, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None,iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    image_res = cv2.bitwise_and(adjusted2, adjusted2, mask=mask)
    cv2.imshow("res", adjusted2)
    cv2.imshow("hsv", image_hsv)
    cv2.imshow("return", image_res)

# 创建对应的自定义的对比度和亮度的调节器
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars", 640, 480)
# cv2.createTrackbar("LH", "TrackBars", 0, 255, hsv)
# cv2.createTrackbar("LS", "TrackBars", 0, 255, hsv)
# cv2.createTrackbar("LV", "TrackBars", 0, 255, hsv)
# cv2.createTrackbar("UH", "TrackBars", 255, 255, hsv)
# cv2.createTrackbar("US", "TrackBars", 255, 255, hsv)
# cv2.createTrackbar("UV", "TrackBars", 255, 255, hsv)
# cv2.createTrackbar("NBrightness", "TrackBars", 0, 256, hsv)
# cv2.createTrackbar("NContrast", "TrackBars", 0, 256, hsv)
# cv2.createTrackbar("PBrightness", "TrackBars", 0, 256, hsv)
# cv2.createTrackbar("PContrast", "TrackBars", 0, 256, hsv)
#
# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()

# adjusted = imutils.adjust_brightness_contrast(image, contrast=80.0, brightness=-180.0)
# if(args['enhance']):
#     adjusted = imutils.adjust_brightness_contrast(adjusted, contrast=50.0, brightness=35.0)
# cv2.imshow("ad", adjusted)
# cv2.waitKey(0)

# cv2.imshow("original", image)
# cv2.waitKey(0)

# cv2.imshow("image_res", image_res)
# cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 1)  # 对原始图像中的一些小的边界进行模糊操作
# gray = cv2.medianBlur(gray, 5)  # 对原始图像中的一些小的边界进行模糊操作
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.imwrite('./gray_images/gray_jiazhuangxian.jpg', gray)


# 使用Canny进行相应的边缘检测
edged = cv2.Canny(gray, 70, 150)
# edged = imutils.auto_canny(gray, sigma=0.8)
# cv2.imshow("edged1", edged)
# cv2.waitKey(0)

edged = cv2.dilate(edged, (5, 5), iterations=1)
edged = cv2.erode(edged, (5, 5), iterations=1)
# cv2.imshow("edged2", edged)
# cv2.waitKey(0)

edged = cv2.GaussianBlur(edged, (7, 7), 0)  # 对边缘进行模糊操作使得边界的粒度更大一些，而不是仅仅关心于局部的边界情况
# edged = cv2.medianBlur(edged, 5)     # 对边缘进行模糊操作使得边界的粒度更大一些，而不是仅仅关心于局部的边界情况
cv2.imshow("edged", edged)
cv2.waitKey(0)

# 在经过边缘检测的图像中进行轮廓特征的提取
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 不同版本的opencv对应的contours的位置是不一致的

# 对轮廓按照从左到右的顺序进行排序操作
(cnts, _) = contours.sort_contours(cnts)  # 对轮廓按照从左到右的顺序进行相应的排序操作

# 用于存储每个像素点对应真实值的比例
pixelsPerMetric = None

# 对每一个轮廓进行单独的处理
for c in cnts:
    final = np.zeros(image.shape, np.uint8)  # 建立对应的最后颜色提取图
    mask = np.zeros(gray.shape, np.uint8)  # 提取对应轮廓的蒙皮
    # 如果轮廓的周长或者面积是不足够大的情况下不予考虑
    if abs(cv2.arcLength(c, closed=True)) < 200:
        continue

    # 计算与轮廓相切的最小的外切矩形
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    # loop over the original points and draw them
    for (x, y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    # 计算顶边和底边的两个中心点
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # 计算左边和右边的两个中心点
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # 计算外切矩形正中心位置处颜色的值
    points = []
    (middleX, middleY) = midpoint([tlblX, tlblY], [trbrX, trbrY])
    point1 = midpoint([tlblX, tlblY], [middleX, middleY])
    point2 = midpoint([tltrX, tltrY], [middleX, middleY])
    point3 = midpoint([trbrX, trbrY], [middleX, middleY])
    point4 = midpoint([blbrX, blbrY], [middleX, middleY])
    points.append(point1)
    points.append(point2)
    points.append(point3)
    points.append(point4)
    points = np.array(points)
    points = np.expand_dims(points, 1)
    print("points.shape: ", points.shape)

    color = orig[int(middleY), int(middleX)]
    color = color.tolist()
    print(color)

    # 绘制每一条边的中心点
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

    # 绘制中点之间的线段
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
             (255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
             (255, 0, 255), 2)

    # 对应中点的连线来作为对应的长和宽的值来使用
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    sample_dist = dist.euclidean(sample_A, sample_B)
    print("dA: {:.2f}, dB: {:.2f}".format(dA, dB))

    pixelsPerMetric = sample_dist / args["width"]
    print("pixelsPerMetric: ", pixelsPerMetric)

    # 获取相应参考物的比例尺
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / args["width"]  # 计算对应的比例尺
        print(pixelsPerMetric)

    # 计算物体的实际尺寸
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    res = cv2.arcLength(c, True)  # 计算对应的轮廓的弧长
    approx = cv2.approxPolyDP(c, 0.001 * res, True)
    ret = cv2.drawContours(orig, [c], -1, (50, 0, 212), 5)
    area = cv2.contourArea(approx)  # 计算对应的轮廓的面积

    # 绘制对应外接矩形的长和宽在相应的图像上
    cv2.putText(orig, "height: {:.2f}cm".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 0, 0), 2)
    cv2.putText(orig, "width: {:.2f}cm".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 0, 0), 2)

    # 提取对应轮廓的颜色均值
    mask[...] = 0
    cv2.drawContours(mask, [c], -1, 255, -1)
    cv2.drawContours(final, [c], -1, cv2.mean(orig, mask), -1)
    mean_color = cv2.mean(orig, mask)

    # 在图像上绘制对应的关键文本信息
    cv2.putText(orig, "arcLength:{:.1f}mm".format(res), (int(trbrX + 30), int(trbrY + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (122, 255, 255), 2)
    cv2.putText(orig, "area:{:.2f}".format(dimA * dimB), (int(trbrX + 30), int(trbrY + 40)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (122, 255, 255), 2)
    cv2.putText(orig, "color: B{}, G{}, R{}".format(color[0], color[1], color[2]), (int(trbrX + 30), int(trbrY + 60)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (122, 255, 255), 2)
    cv2.rectangle(orig, (0, 0), (40, 40), (mean_color[0], mean_color[1], mean_color[2]), -1)
    cv2.putText(final, "color: B{}, G{}, R{}".format(int(mean_color[0]), int(mean_color[1]), int(mean_color[2])),
                (int(trbrX + 30), int(trbrY + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (122, 255, 255), 2)

    # 展示对应的图像
    cv2.imshow("Image", orig)
    cv2.imshow("mask", mask)
    cv2.imshow("final", final)
    cv2.waitKey(0)
