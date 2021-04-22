# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import imutils
import cv2
from PIL import Image
from predict import *

# 计算两点之间的距离
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# 计算直尺的宽度比例,主要用于比例尺的计算工作
def process_contours_zc(cnts_zc, pixelsPerMetric):

    for c in cnts_zc:
        # 出于鲁棒性的考量， 直尺的周长应该更长
        if abs(cv2.arcLength(c, closed=True)) < 400:
            return

        # 计算与轮廓相切的最小的外切矩形
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        # 对轮廓中的点进行左上，右上， 右下， 左下的排序操作
        box = perspective.order_points(box)

        # 计算顶边和底边的两个中心点
        (tl, tr, br, bl) = box

        top_dist = dist.euclidean(tl, tr)  # 顶边的长度
        left_dist = dist.euclidean(tl, bl)  # 侧边的长度

        if top_dist >= left_dist:  # 直尺是立着的
            dA = left_dist
        else:  # 直尺是横着的
            dA = top_dist

        # 获取相应参考物的比例尺
        if pixelsPerMetric is None:
            pixelsPerMetricF = dA / 2.7  # 计算对应的比例尺
            print("根据直尺首次计算出的比例尺: ", pixelsPerMetricF)
            return pixelsPerMetricF

        return None

# 计算样本的轮廓
def process_contours(cnts, image, pixelsPerMetric):
    # 对每一个轮廓进行单独的处理
    for c in cnts:
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

        # 获取相应参考物的比例尺
        if pixelsPerMetric is None:
            sample_dist = dist.euclidean(sample_A, sample_B)
            print("dA: {:.2f}, dB: {:.2f}".format(dA, dB))
            pixelsPerMetric = sample_dist / width
            print("pixelsPerMetric is None: ", pixelsPerMetric)
            pixelsPerMetric = dB / 2.3  # 计算对应的比例尺
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
        cv2.putText(orig, "arcLength:{:.1f}mm".format(res), (int(trbrX + 30), int(trbrY + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (122, 255, 255), 2)
        cv2.putText(orig, "area:{:.2f}".format(dimA * dimB), (int(trbrX + 30), int(trbrY + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (122, 255, 255), 2)
        cv2.putText(orig, "color: B{}, G{}, R{}".format(color[0], color[1], color[2]),
                    (int(trbrX + 30), int(trbrY + 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (122, 255, 255), 2)
        # cv2.rectangle(orig, (0, 0), (40, 40), (mean_color[0], mean_color[1], mean_color[2]), -1)
        # cv2.putText(final, "color: B{}, G{}, R{}".format(int(mean_color[0]), int(mean_color[1]), int(mean_color[2])),
        #             (int(trbrX + 30), int(trbrY + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (122, 255, 255), 2)

        # 展示对应的图像
        cv2.imshow("Image", orig)
        # cv2.imshow("mask", mask)
        # cv2.imshow("final", final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 定义对应的命令行参数
args = get_args()
width = args.width

# 作为备用点计算，防止没有放置直尺。或者直尺没有捕捉到
sample_A = (138, 681)
sample_B = (179, 682)

# 使用predict进行mask的提取

net = UNet(n_channels=3, n_classes=1)  # 加载模型

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置处理设备

net.to(device=device)
net.load_state_dict(torch.load(args.model, map_location=device))  # 加载模型


net_zc = UNet(n_channels=3, n_classes=1)  # 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置处理设备
net_zc.to(device=device)
net_zc.load_state_dict(torch.load(args.model_zc, map_location=device))  # 加载模型

import glob
input_files = glob.glob("./test_images/*", recursive=True)

cap = cv2.VideoCapture(0)
ret = cap.isOpened()
fps = cap.get(5)/10000  #查询帧率
while ret:

    _, img = cap.read()
    tstep = cap.get(1)
    iloop = fps / 2

    # img = Image.open(fn)
    image = img.copy()
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 转换为Image格式

    # 获得样本的轮廓
    mask, mask_zc = predict_img(net=net, net_zc=net_zc,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.mask_threshold,
                       device=device)


    # 转换为对应的PIL.Image格式
    result = mask_to_image(mask)

    # 转为numpy.array的形式，使得之后的opencv能够进行调用
    mask = np.asarray(result)

    #使用Canny进行相应的边缘检测
    edged = cv2.Canny(mask, 0, 100)  # 样本图像的边缘

    # 对边缘信息进行模糊处理，能够对一些断开的边缘信息起到对应的连接的作用
    edged = cv2.GaussianBlur(edged, (3, 3), 0)  # 对边缘进行模糊操作使得边界的粒度更大一些，而不是仅仅关心于局部的边界情况

    # 在经过边缘检测的图像中进行轮廓特征的提取
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)  # 找到对应的样本的轮廓
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 不同版本的opencv对应的contours的位置是不一致的

    # 用于存储每个像素点对应真实值的比例
    pixelsPerMetric = None

    result_zc = mask_to_image(mask_zc)
    mask_zc = np.asarray(result_zc)
    backup_zc = np.zeros((mask_zc.shape[0] + 10, mask_zc.shape[1] + 10), dtype=np.uint8)
    backup_zc[5:mask_zc.shape[0]+5, 5: mask_zc.shape[1]+5] = mask_zc[:,:]
    mask_zc = backup_zc

    mask_zc = cv2.erode(mask_zc, None, iterations=8)
    mask_zc = cv2.dilate(mask_zc, None, iterations=8)
    edged_zc = cv2.Canny(mask_zc, 0, 100) # 直尺边缘
    # 提取直尺对应的轮廓
    cnts_zc = cv2.findContours(edged_zc.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_zc = cnts_zc[0] if imutils.is_cv2() else cnts_zc[1]

    pixelsPerMetric = process_contours_zc(cnts_zc, pixelsPerMetric)
    # process_contours(cnts_zc, image, pixelsPerMetric)
    process_contours(cnts, image, pixelsPerMetric)  # 对样本的轮廓进行相应的处理操作
    while iloop:
        cap.grab()  # 只取帧不解码，
        iloop = iloop - 1
        if iloop < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()


