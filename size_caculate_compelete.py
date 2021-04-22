# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import cv2
from PIL import Image
from predict import *

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# 定义对应的命令行参数
args = get_args()
width = args.width

sample_A = (138, 681)
sample_B = (179, 682)


# 使用predict进行mask的提取

net = UNet(n_channels=3, n_classes=1)  # 加载模型

logging.info("Loading model {}".format(args.model))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置处理设备
logging.info(f'Using device {device}')

net.to(device=device)
net.load_state_dict(torch.load(args.model, map_location=device))  # 加载模型

logging.info("Model loaded !")

import glob
input_files = glob.glob("./standardSample/*", recursive=True)

for i, fn in enumerate(input_files):
    logging.info("\nPredicting image {} ...".format(fn))

    # img = Image.open(fn)
    img = cv2.imread(fn)  # 读取原始样本
    print(img.shape)
    coin_img = cv2.imread("./stand_coin.jpg")
    imageROI = np.ones((100, 88, 3))
    imageROI = coin_img[0:100, 0:88]

    img[10:98, 10:98] = imageROI

    cv2.imshow("img", img)
    cv2.waitKey(0)
    exit()

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 将opencv的形式转换为PIL.Image

    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       device=device)  # 对原始图像求取mask

    result = mask_to_image(mask)  # 将mask转换为对应的形式

    mask = np.asarray(result)  # 转换为numpy数组
    print("mask shape: ", mask.shape)

    #使用Canny进行相应的边缘检测
    edged = cv2.Canny(mask, 0, 100)  # 对mask图像进行边缘检测


    # 对边缘信息进行模糊处理，能够对一些断开的边缘信息起到对应的连接的作用
    edged = cv2.GaussianBlur(edged, (3, 3), 0)  # 对边缘进行模糊操作使得边界的粒度更大一些，而不是仅仅关心于局部的边界情况

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
        # final = np.zeros(image.shape, np.uint8)  # 建立对应的最后颜色提取图
        # mask = np.zeros(image.shape, np.uint8)  # 提取对应轮廓的蒙皮
        # 如果轮廓的周长或者面积是不足够大的情况下不予考虑
        if abs(cv2.arcLength(c, closed=True)) < 200:
            continue

        # 计算与轮廓相切的最小的外切矩形
        orig = img.copy()
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

        pixelsPerMetric = sample_dist / width
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
        # mask[...] = 0
        # cv2.drawContours(mask, [c], -1, 255, -1)
        # cv2.drawContours(final, [c], -1, cv2.mean(orig, mask), -1)
        # mean_color = cv2.mean(orig, mask)

        # 在图像上绘制对应的关键文本信息
        cv2.putText(orig, "arcLength:{:.1f}mm".format(res), (int(trbrX + 30), int(trbrY + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (122, 255, 255), 2)
        cv2.putText(orig, "area:{:.2f}".format(dimA * dimB), (int(trbrX + 30), int(trbrY + 40)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (122, 255, 255), 2)
        cv2.putText(orig, "color: B{}, G{}, R{}".format(color[0], color[1], color[2]), (int(trbrX + 30), int(trbrY + 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (122, 255, 255), 2)
        # cv2.rectangle(orig, (0, 0), (40, 40), (mean_color[0], mean_color[1], mean_color[2]), -1)
        # cv2.putText(final, "color: B{}, G{}, R{}".format(int(mean_color[0]), int(mean_color[1]), int(mean_color[2])),
        #             (int(trbrX + 30), int(trbrY + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (122, 255, 255), 2)

        # 展示对应的图像
        cv2.imshow("Image", orig)
        # cv2.imshow("mask", mask)
        # cv2.imshow("final", final)
        cv2.waitKey(0)
