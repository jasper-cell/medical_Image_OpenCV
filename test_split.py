import cv2
import numpy as np
from skimage import morphology

img1 = cv2.imread('./standardSample/zigongjiliu.jpg')
cv2.imshow('img1',img1)
#自适应阈值分割
gray= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,3)
cv2.imshow('img2',img2)
#反色
def inverse_color(img):
    height,width = img.shape
    img2 = img.copy()
    for i in range(height):
        for j in range(width):
            img2[i,j] = (255-img[i,j])
    return img2
img3 = inverse_color(img2)
cv2.imshow('img3',img3)
#对图像进行扩展
img4 = cv2.copyMakeBorder(img3,1,1,1,1,cv2.BORDER_REFLECT)
cv2.imshow('img4',img4)

#去除小于指定尺寸的区域
img5 = morphology.remove_small_holes(img2,150)
img_tmp1 = np.uint8(img5)*255
cv2.imshow('img5',inverse_color(img_tmp1))
img6 = morphology.remove_small_holes(img5,1000)
img6 = np.uint8(img6)*255
img7 = inverse_color(img6)
cv2.imshow('img6',inverse_color(img6))

#图像细化
img = img7.copy()
img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img=cv2.adaptiveThreshold(gray,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,3)
img_sk = morphology.skeletonize(img)
img = np.uint8(img_sk)*255
cv2.imshow('img8',inverse_color(img))

#边界提取
img9 = cv2.Canny(inverse_color(img6),75,200)
cv2.imshow('img9',img9)
cv2.imshow('img10',inverse_color(img9))
cv2.waitKey()
cv2.destroyAllWindows()