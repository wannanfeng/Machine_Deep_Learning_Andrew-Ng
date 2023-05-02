from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn import svm
def plot_origin_data():
    posx = x[np.where(y==1)]
    negx = x[np.where(y==0)]
    plt.scatter(posx[:,0],posx[:,1],c='r')
    plt.scatter(negx[:,0],negx[:,1],c='b')
def get_best_params(x,y,xval,yval):
    best_score = 0 #存放最佳准确率
    best_params = (0,0) #存放最佳参数C and σ
    for c in Cvalues:
        for gamma in gammas:
            svc1 = svm.SVC(C=c,kernel='rbf',gamma=gamma)
            svc1.fit(x,y)
            score = svc1.score(xval,yval)
            if score>best_score:
                best_score =  score
                best_params = (c,gamma)
    return best_score,best_params
def plot_boundary(model):
    x_min , x_max = -0.6,0.4
    y_min , y_max = -0.7,0.6
    x = np.linspace(x_min,x_max,500)
    y = np.linspace(y_min,y_max,500)
    xx,yy = np.meshgrid(x,y)
    z = model.predict(np.c_[xx.flatten(),yy.flatten()])
    zz = z.reshape(xx.shape)
    plt.contour(xx,yy,zz)
    plot_origin_data()
    plt.show()
data = loadmat('ex6data3.mat')
x = data['X']
y = data['y'].flatten()
xval = data['Xval']
yval = data['yval'].flatten()
# -0.596774 0.297235 -0.657895 0.573392 ; choose (-0.6,0.4) (-0.7,0.6)
# print(np.min(x[:,0]),np.max(x[:,0]),np.min(x[:,1]),np.max(x[:,1]))
#设定9个c和gamma的值来对比各个值的拟合度
Cvalues = [3, 10, 30, 100, 0.01, 0.03, 0.1, 0.3, 1]
gammas = [1, 3, 10, 30, 100, 0.01, 0.03, 0.1, 0.3]

best_score , best_param = get_best_params(x,y,xval,yval)
best_svc = svm.SVC(C=best_param[0],kernel='rbf',gamma=best_param[1])
best_svc.fit(x,y)
print(best_score,best_param)
plt.figure()
plot_boundary(best_svc)