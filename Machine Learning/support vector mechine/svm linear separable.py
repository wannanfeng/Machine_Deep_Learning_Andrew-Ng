from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn import svm
def pltshow(): #绘制初始数据
    posx = x[np.where(y==1)]
    negx = x[np.where(y==0)]
    plt.scatter(posx[:,0],posx[:,1],c='r')
    plt.scatter(negx[:,0],negx[:,1],c='b')

def plt_brounder(model): #传入模型绘制决策边界
    x_min,x_max = -0.5,4.5
    y_min,y_max = 1.3,5
    # 生成网格点坐标矩阵
    xx,yy = np.meshgrid(np.linspace(x_min,x_max,500),np.linspace(y_min,y_max,500))
    z = model.predict(np.c_[xx.flatten(),yy.flatten()]) # z (250000,)
    #np.c_ 按行连接两矩阵，np.r_ 按列连接矩阵
    zz = z.reshape(xx.shape) #将预测出来的z值有一维空间，转为二维网格坐标矩阵，便于在二维平面绘制决策边界
    plt.contour(xx,yy,zz) #对网格中每个点的值等于一系列值的时候做出一条条轮廓线，类似于等高线 。
    pltshow()
    plt.show()

data = loadmat("ex6data1.mat")
x = data['X']
y = data['y'].flatten()
#获取横纵最值，由此得出横轴取值范围为（-0.5，4.5），竖轴取值范围为（1.3，5）
# print(np.min(x[:,0]),np.max(x[:,0]))    #0.086405 4.015
# print(np.min(x[:,1]),np.max(x[:,1]))    #1.6177 4.6162
c = 1 #误差惩罚系数,过大或小会导致过拟合或欠拟合
svc1 = svm.SVC(C=c,kernel='linear') #创建一个分类器 核函数为线性的
svc1.fit(x,y) #传入需要拟合的数据,训练svc1这训练器
plt_brounder(svc1)
# print(svc1.predict(x))#用这分类器预测新数据
# print(svc1.score(x,y)) #计算预测得分