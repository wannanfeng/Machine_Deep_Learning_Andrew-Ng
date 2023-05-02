import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.cluster import KMeans

image_data = sio.loadmat("bird_small.mat")
data = image_data['A']
#数据归一化
data = data / 255
X = np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2]))

model = KMeans(n_clusters=16,n_init=100)  #n_init设置获取初始簇中心的更迭次数，防止局部最优 n_jobs设置并行（使用CPU数，-1则使用所有CPU）
model.fit(X)    #开始聚类

centroids = model.cluster_centers_  #获取聚簇中心
C = model.predict(X) #获取每个数据点的对应聚簇中心的索引

X_recovered = centroids[C].reshape((data.shape[0],data.shape[1],data.shape[2])) #获取新的图像

plt.figure()
plt.imshow(data)    #显示原始图像
plt.show()

plt.figure()
plt.imshow(X_recovered) #显示压缩后的图像
plt.show()