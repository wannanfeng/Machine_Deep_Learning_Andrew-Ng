import sys

from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plt_origin_data():
    plt.figure()
    plt.scatter(x[:,0],x[:,1])
    plt.show()
def random_init_centroids(x,k):
    centroids = np.zeros((k,x.shape[1]))
    index = np.random.randint(0,x.shape[0],k) #从0到x.shape[0]取出k个返回数组
    centroids = x[index,:]
    return centroids
def find_closest_centroids(x,centroids):
    index = np.zeros(len(x))
    index = index.astype(int)
    for i in range(len(x)):

        # index[i] = np.sum(np.power((x[i]-centroids),2),axis=1) #每个点和聚类中心点的距离 此行报错，以下是chatGPT解决方法，nice的
        # 请注意您的代码中的语句
        # index[i] = np.sum(np.power((x[i] - centroids), 2), axis=1)，它尝试在数组
        # index的第i个位置上放置一个值，但是np.sum()的返回值是一个具有两个元素的一维数组，而不是一个标量。因此，您需要选择一个要放入
        # index[i]的元素。例如，您可以使用以下语句：index[i] = np.argmin(np.sum(np.power((x[i] - centroids), 2), axis=1))
        # 在此语句中，np.argmin()选择了np.sum(np.power((x[i] - centroids), 2), axis=1)中最小的元素的索引，并将其存储在index[i]中。
        index[i] = np.argmin(np.sum(np.power((x[i]-centroids),2),axis=1)) #返回离最近中心的索引，即把该点归属于该中心

    return index
def update_centroids(x,index,k):
    centroids_new = []
    for i in range(k):
        centroids_i = np.mean(x[np.where(index==i)],axis=0)
        centroids_new.append(centroids_i)
    return np.array(centroids_new)
def run_kmeans(x,centroids):
    all_centroids = []
    all_centroids.append(centroids) #加入初始中心点
    iters = 0 #统计迭代次数
    while True: #直到出现连续相同位置的中心点
        index = find_closest_centroids(x,centroids)
        centroids = update_centroids(x,index,K)
        all_centroids.append(centroids)
        iters = iters + 1
        if (all_centroids[iters-1]==all_centroids[iters]).all(): #需要理解
            del all_centroids[iters]
            break

    return index,np.array(all_centroids),iters-1
def plt_show():
    cluster_1 = x[np.where(index == 0)]
    cluster_2 = x[np.where(index == 1)]
    cluster_3 = x[np.where(index == 2)]
    plt.figure()
    plt.scatter(cluster_1[:, 0], cluster_1[:, 1], c='r', marker="o")
    plt.scatter(cluster_2[:, 0], cluster_2[:, 1], c='b', marker="o")
    plt.scatter(cluster_3[:, 0], cluster_3[:, 1], c='g', marker="o")
    plt.plot(all_centroids[:, :, 0], all_centroids[:, :, 1], c='black', marker='x')
    plt.show()


data = loadmat('ex7data2.mat')
x = data['X']

# print(x.shape) #(300,2)
K = 3 #设置3个簇

initial_centroids = random_init_centroids(x,K)

index,all_centroids,iters = run_kmeans(x,initial_centroids)
plt_show()
# print(all_centroids)
# print(all_centroids[:,:,0]) #输出所有中心的x坐标
print("Final iters:",iters)


#test np.sum(,axis=???)
# k = np.array([[1,1],[2,3],[4,3]])
# print(x[1]-k)
# for i in range(len(x)):
#     print(np.sum(np.power(x[i]-k,2)))
#     print(np.sum(np.power(x[i]-k,2),axis=1))
#     print(np.sum(np.power(x[i] - k, 2), axis=0))
#     print('wwwwwwwwwwwwwwwwwwwwww')


