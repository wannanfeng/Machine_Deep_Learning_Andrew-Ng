from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn import svm
def gaussian_kernel(x1,x2,sigma=0.1): #手写 高斯核函数
    x1 = x1.flatten()
    x2 = x2.flatten()
    return np.exp(-np.sum(np.power(x1-x2),2)/2/np.power(sigma,2))
def plt_origin_data():
    posx = x[np.where(y==1)]
    negx = x[np.where(y==0)]
    plt.scatter(posx[:,0],posx[:,1],c='r')
    plt.scatter(negx[:,0],negx[:,1],c='b')

def plt_brounder(model): #传入模型绘制决策边界
    x_min,x_max = 0,1
    y_min,y_max = 0.4,1
    # 生成网格点坐标矩阵
    xx,yy = np.meshgrid(np.linspace(x_min,x_max,500),np.linspace(y_min,y_max,500))
    z = model.predict(np.c_[xx.flatten(),yy.flatten()]) # z (250000,)
    #np.c_ 按行连接两矩阵，np.r_ 按列连接矩阵
    zz = z.reshape(xx.shape) #将预测出来的z值有一维空间，转为二维网格坐标矩阵，便于在二维平面绘制决策边界
    plt.contour(xx,yy,zz) #对网格中每个点的值等于一系列值的时候做出一条条轮廓线，类似于等高线 。
    plt_origin_data()
    plt.show()

data = loadmat("ex6data2.mat")
x = data['X']
y = data['y'].flatten()
# print(x.shape,y.shape)
# print(np.min(x[:,0]),np.max(x[:,0]))    #0.0449309 0.998848 取横坐标从（0，1）,竖轴（0.4，1）
# print(np.min(x[:,1]),np.max(x[:,1]))    #0.402632 0.988596
#调用svm中高斯核函数
c = 1
svc1 = svm.SVC(C=c,kernel='rbf',gamma=50) #rbf为高斯核函数，gamma为高斯核函数分子的参数 ,gamma=1/(2*σ*σ)??
svc1.fit(x,y) #gamma越大,即σ小，模型越复杂;越小则σ大，模型不复杂，即为类似过拟合和非过拟合
print(svc1.score(x,y))
plt_brounder(svc1)
#可尝试改变c 或者 gamma值来得到不同的拟合程度