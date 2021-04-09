import numpy as np
import cv2
import random
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as mp

img = cv2.pyrDown(cv2.imread("C:/Users/Skyvein/Desktop/cmera_cross/zhuo_ban_test/DSC00002.JPG", cv2.IMREAD_UNCHANGED))  # 使用下采样的方法

ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY)  # 对图像进行二值化处理
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 对图像中的轮廓进行捕捉
contours = sorted(contours, key=cv2.contourArea, reverse=True)  # 对图像中捕捉到的轮廓进行按照面积的大小进行排序
ys_list = contours[0]
cv2.drawContours(img, contours[0], -1, (255, 0, 0), 2)
ball_list = []
print(len(ys_list))
while (len(ys_list) > 0):

    temp_list = []
    seed = random.randint(0,len(ys_list)-1)
    temp_list.append(ys_list[seed][0])

    temp_end = np.delete(ys_list, seed, axis = 0)
    i_list = []
    for i in range(len(temp_end)):
        if (abs(temp_end[i][0][0] - ys_list[seed][0][0]) < 20).all() and (abs(temp_end[i][0][1] - ys_list[seed][0][1]) < 20).all():
            temp_list.append(temp_end[i][0])
            i_list.append(i)
        else:
            pass
    for j in i_list[::-1]:
        temp_end = np.delete(temp_end, j, axis = 0)
    ball_list.append(temp_list)
    ys_list = temp_end
ff = []
while(len(ball_list) > 0):
    temp_lis = []
    temp_lis.append(ball_list[0])

    temp_ll = np.delete(ball_list, 0, axis = 0)
    opend = True
    while opend:
        tem = False
        for i in range(len(temp_lis)):
            j_list = []
            for j in range(len(temp_ll)):
                if np.sqrt(np.sum((temp_lis[i][0]-temp_ll[j][0])**2)) < 20*np.sqrt(20):
                    temp_lis.append(temp_ll[j])
                    j_list.append(j)
                    tem = True
            if len(j_list) > 0:
                for k in j_list[::-1]:
                    temp_ll = np.delete(temp_ll, k, axis = 0)
            if len(temp_lis) - 1 == i and tem == False:
                opend = False
    gg_list = []
    for tt in temp_lis:
        for dd in tt:
            gg_list.append([dd])
    ff.append(gg_list)
    ball_list = temp_ll

ff = sorted(ff, key=lambda x:len(x), reverse=True)
rect = cv2.minAreaRect(np.array(ff[0]))
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),3)


# cv2.drawContours(img, contours[0], -1, (255, 0, 0), 2)
cv2.namedWindow('thresh',0)
cv2.resizeWindow('thresh',640,480)
cv2.imshow("thresh", img)
cv2.waitKey(0)