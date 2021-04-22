from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse

gamma = 1.0
gamma_max = 400


def gammaCorrection():
    ## [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv.LUT(img_original, lookUpTable)
    ## [changing-contrast-brightness-gamma-correction]

    img_gamma_corrected = cv.hconcat([img_original, res])
    cv.imshow("Gamma correction", img_gamma_corrected)


parser = argparse.ArgumentParser(description='Code for Changing the contrast and brightness of an image! tutorial.')
parser.add_argument('--input', help='Path to input image.', default='./standardSample/ruxian.jpg')
args = parser.parse_args()

# 读取对应的图像
img_original = cv.imread(cv.samples.findFile(args.input))
if img_original is None:
    print('Could not open or find the image: ', args.input)
    exit(0)

# 对gamma值进行初始化
gamma_init = int(gamma * 200)

## [changing-contrast-brightness-gamma-correction]
lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, 1.82) * 255.0, 0, 255)

res = cv.LUT(img_original, lookUpTable)

cv.imshow("Gamma correction", res)
cv.waitKey()