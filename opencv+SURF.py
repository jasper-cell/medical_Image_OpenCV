import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#img = cv2.imread("C:/Users/Skyvein/Desktop/cur_dataset/zhuo_ban_test/DSC00034.JPG")
img = cv2.pyrDown(cv2.imread("C:/Users/Skyvein/Desktop/cmera_cross/zhuo_ban_test/DSC00037.JPG", cv2.IMREAD_UNCHANGED))

ret, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) , 127, 255, cv2.THRESH_BINARY)
contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
llist = []
for cont in contours[0]:
    llist.append(cont[0].tolist())

def find_index(arr, data):
    ind = 0
    fin = 0
    for aa in arr:
        if (data == aa).all():
            fin = ind
        ind += 1
    return fin

def dis(m,n):
    return np.sqrt(np.sum((m-n)**2))
llist = np.array(llist)
dad_list = []
while(len(llist) > 0):
    temp_list = []
    temp_list.append(llist[0])

    temp_ll = np.delete(llist, 0, axis = 0)
    opend = True
    while opend:
        temp = False
        for i in range(len(temp_list)):
            #llis = np.delete(llist, i, axis = 0)
            temp_dis_lsit = []
            for j in range(len(temp_ll)):
                temp_dis = dis(temp_list[i],temp_ll[j])
                temp_dis_lsit.append(temp_dis)
            A = list(zip(temp_dis_lsit,temp_ll))
            
            min_dis = sorted(A,key=lambda x :x[0],reverse=False)

            for k in range(len(min_dis)):
                if int(min_dis[k][0]) < 2000:
                    temp_list.append(min_dis[k][1])
                    index = find_index(temp_ll, min_dis[k][1])
                    temp_ll = np.delete(temp_ll,index,axis=0)
                    temp = True
            if i == len(temp_list) - 1 and temp == False:
                opend = False
    dad_list.append(temp_list)
    llist = temp_ll
    print(len(llist))

print(len(dad_list))
print(dad_list)
    








'''
def localoutlierfactor(data, predict, k):
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1)
    clf.fit(data)
    # 记录 k 邻域距离
    predict['k distances'] = clf.kneighbors(predict)[0].max(axis=1)
    # 记录 LOF 离群因子，做相反数处理
    predict['local outlier factor'] = -clf._decision_function(predict.iloc[:, :-1])
    return predict

def plot_lof(result, method):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(8, 4)).add_subplot(111)
    plt.scatter(result[result['local outlier factor'] > method].index,
                result[result['local outlier factor'] > method]['local outlier factor'], c='red', s=50,
                marker='.', alpha=None,
                label='离群点')
    plt.scatter(result[result['local outlier factor'] <= method].index,
                result[result['local outlier factor'] <= method]['local outlier factor'], c='black', s=50,
                marker='.', alpha=None, label='正常点')
    plt.hlines(method, -2, 2 + max(result.index), linestyles='--')
    plt.xlim(-2, 2 + max(result.index))
    plt.title('LOF局部离群点检测', fontsize=13)
    plt.ylabel('局部离群因子', fontsize=15)
    plt.legend()
    plt.show()

def lof(data, predict=None, k=5, method=1, plot=False):
    import pandas as pd
    # 判断是否传入测试数据，若没有传入则测试数据赋值为训练数据
    try:
        if predict == None:
            predict = data.copy()
    except Exception:
        pass
    predict = pd.DataFrame(predict)
    # 计算 LOF 离群因子
    predict = localoutlierfactor(data, predict, k)
    if plot == True:
        plot_lof(predict, method)
    # 根据阈值划分离群点与正常点
    outliers = predict[predict['local outlier factor'] > method].sort_values(by='local outlier factor')
    inliers = predict[predict['local outlier factor'] <= method].sort_values(by='local outlier factor')
    return outliers, inliers
lat = []
lon = []
for c in contours[0]:
    lat.append(c[0][0])
    lon.append(c[0][1])
lat = np.array(lat)
lon = np.array(lon)
A = list(zip(lat, lon))  # 按照纬度-经度匹配

# # 获取任务密度，取第5邻域，阈值为2（LOF大于2认为是离群值）
for k in [3,5,10]:
    plt.figure('k=%d'%k)
    outliers1, inliers1 = lof(A, k=k, method = 2)
    plt.scatter(np.array(A)[:,0],np.array(A)[:,1],s = 10,c='b',alpha = 0.5)
    plt.scatter(outliers1[0],outliers1[1],s = 10+outliers1['local outlier factor']*100,c='r',alpha = 0.2)
    plt.title('k=%d' % k)
    plt.show()

'''

#######################################################################

# cv2.drawContours(img, contours[0], -1, (255, 0, 0), 2)
# cv2.namedWindow('thresh',0)
# cv2.resizeWindow('thresh',640,480)
# cv2.imshow("thresh", img)

# # cv2.drawContours(img, contours, -1, (255, 0, 255), 2)   #绘制边沿轮廓 
# cv2.waitKey()
# cv2.destroyAllWindows()



#####################################################################

# ls = cv2.HoughLinesP(edges,1,np.pi/180,10,lines=10,minLineLength=20)
# l = ls[:,0,:]

# for x1,y1,x2,y2 in l[:]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),10)

# cv2.namedWindow('img',0)
# cv2.resizeWindow('img',640,480)
# cv2.imshow('img',img)

# cv2.namedWindow('gray',0)
# cv2.resizeWindow('gray',640,480)
# cv2.imshow('gray',edges)

# cv2.waitKey(0)