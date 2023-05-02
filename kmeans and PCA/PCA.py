import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import warnings
warnings.filterwarnings("ignore")

data = loadmat("ex7data1.mat")
# print(data.keys())
x = data['X']
x = x - np.mean(x,axis=0) #取均值化

c = x.T.dot(x)/len(x)#协方差矩阵
# print(c)
U,S,V = np.linalg.svd(c) #线性代数库里面的svd
# print(U,S) #降成一维所以取U第一列
U1 = U[:,:1] #attention --------- 与U[:,0]获得的数据维度有区别

z_down = x.dot(U1)#对原始数据降维
#要在二维上显示，得升维回去
z_up = z_down.dot(U1.T)



plt.figure(figsize=(6,6))
plt.scatter(x[:,0],x[:,1])
plt.scatter(z_up[:,0],z_up[:,1],c='pink')
plt.plot([0,U1[0]],[0,U1[1]],c='r') #投影的方向向量
for i in range(len(x)):
    plt.plot([x[i][0],z_up[i][0]],[x[i][1],z_up[i][1]],linestyle='--',c='black')

plt.plot([0,U[:,1][0]],[0,U[:,1][1]],c='k') #与上个方向正交

plt.show()