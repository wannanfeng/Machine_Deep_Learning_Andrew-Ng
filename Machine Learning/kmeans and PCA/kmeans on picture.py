import sys

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

#使用K-Means进行图像压缩
#我们的特征就是颜色空间三通道，所以我们后面求取的聚类中心就是我们找到的最具代表的颜色空间
def find_closest_centroids(X,centroids):
    m = X.shape[0]
    idx = np.zeros(m)   #记录每个训练样本距离最短聚类中心最短的索引
    idx = idx.astype(int)   #因为numpy中没有int、float类型，是由系统决定是32、或者64位大小。所以我们这里手动设置位int类型，为后面做准备
    for i in range(m):
        idx[i] = np.argmin(np.sum(np.power((centroids-X[i]),2),axis=1))  #先计算各个中心到该点的平方和距离，返回最小的索引

    return idx

def compute_centroids(X,idx,K):
    (m,n)=X.shape
    centroids_new = np.zeros((k,n))

    #进行更新操作，用每个聚类中心所有点的位置平均值作为新的聚类中心位置
    for i in range(K):
        centroids_new[i] = np.mean(X[np.where(idx==i)[0]],axis=0)    #按列求均值

    return centroids_new

def run_k_means(X,init_centroids,max_iters=0):
    m,n = X.shape
    idx = np.zeros(m)
    k = init_centroids.shape[0]
    centroids = init_centroids

    #开始迭代
    if max_iters != 0:
        for i in range(max_iters):  #按迭代次数进行迭代
            idx = find_closest_centroids(X,centroids)
            centroids = compute_centroids(X,idx,k)
    else:
        while True: #直到连续两次的迭代结果都是一样的，就返回
            idx = find_closest_centroids(X, init_centroids)
            centroids = compute_centroids(X,idx,k)
            if (init_centroids == centroids).all():
                break
            init_centroids = centroids

    return idx,centroids

def kmeans_init_centroids(X,k):
    centroids = np.zeros((k,X.shape[1]))

    #随机选取训练样本个数
    idx = np.random.choice(X.shape[0],k)
    centroids = X[idx,:]

    return centroids

image_data= loadmat('bird_small.mat')
# print(image_data.keys())
init_data = image_data['A'] #RGB范围0-255
# print(data,data.shape)
data = init_data/255 #归一化
X = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]))
k = 16
max_iters = 10

#随机初始化聚类中心
init_centroids = kmeans_init_centroids(X,k)
#获取聚类中心
idx,centroids = run_k_means(X,init_centroids,max_iters)
#将所有数据点，设置归属到对应的聚类中心去
idx = find_closest_centroids(X,centroids)
#将每一个像素值与聚类结果进行匹配
X_recovered = centroids[idx,:]  #将属于一个聚类的像素，设置为聚类中心的值（统一）
# print(X_recovered.shape)    #(16384, 3)
X_recovered = np.reshape(X_recovered,(data.shape[0],data.shape[1],data.shape[2]))  #再展开为三维数据

plt.figure()
plt.imshow(init_data)    #显示原始图像
plt.show()

plt.figure()
plt.imshow(X_recovered) #显示压缩后的图像
plt.show()