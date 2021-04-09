import cv2
import numpy as np
import glob

# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# 获取标定板角点的位置
objp = np.zeros((8 * 11, 3), np.float32)
objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
print(objp.shape)

obj_points = []  # 存储3D点
img_points = []  # 存储2D点

images = glob.glob("C:\\Users\\Skyvein\\Desktop\\cmera_cross\\biaoding\\*.jpg", recursive=True)  # 在路径中不应该存在中文路径
print(len(images))

i=0
for fname in images:
    img = cv2.imread(fname)   # 读取图像到内存
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    size = gray.shape[::-1]  # 对灰度图像的高和宽进行转置并对其尺寸进行记录
    ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)  # 获得棋盘角点的位置, ret 为bool类型记录对应的函数是否成功的返回。 corners为角点的数据矩阵
    # print(corners)
    # cv2.drawChessboardCorners(img, (11, 8), corners, ret)
    # cv2.imshow('img',images)
    # cv2.waitKey(0)
    if ret:  # 如果对应的函数执行成功的话，继续执行以下操作

        obj_points.append(objp)  # 将对应的3D点置入对应的列表中

        corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
        #print(corners2)
        if [corners2]:
            img_points.append(corners2)
        else:
            img_points.append(corners)

        # 绘制对应的角点在棋盘对应的位置上， 并对其进行保存操作
        cv2.drawChessboardCorners(img, (11, 8), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
        i+=1
        cv2.imwrite('C:/Users/Skyvein/Desktop/cmera_cross/save_test_data/'+str(i)+'.jpg', img)
        cv2.waitKey(1500)

print(len(img_points))
cv2.destroyAllWindows()

# exit()
# 标定
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)

print("ret:", ret)
print("mtx:\n", mtx) # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs ) # 平移向量  # 外参数

print("-----------------------------------------------------")

img = cv2.imread('C:/Users/Skyvein/Desktop/cmera_cross/zhuo_ban_test/DSC00009.JPG')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#显示更大范围的图片（正常重映射之后会删掉一部分图像）
print (newcameramtx)
print("------------------使用undistort函数-------------------")
dst = cv2.undistort(img,mtx,dist,None,newcameramtx)
x,y,w,h = roi
dst1 = dst[y:y+h,x:x+w]
cv2.imwrite('C:/Users/Skyvein/Desktop/cmera_cross/biaoding_res/calibresult3.jpg', dst1)
print ("方法一:dst的大小为:", dst1.shape)
