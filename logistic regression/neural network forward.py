#手写识别数字
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat    #用于载入matlab文件
from scipy.optimize import minimize
k = 10
lamda = 1
def sigmoid(z):
    return 1/(1+np.exp(-z))
def costfunction(theta,x,y,lamda):
    A = sigmoid(x.dot(theta))
    first = np.multiply(y,np.log(A))
    last = np.multiply(1-y,np.log(1-A))
#    reg = np.sum(np.power(theta[1:],2))*lamda/2/len(x)
    reg = theta[1:].dot(theta[1:])*lamda/2/len(x)
    return -np.sum(first+last)/len(x)+reg
def gradient(theta,x,y,lamda): #梯度
    reg = theta[1:]*lamda/len(x)
    reg = np.insert(reg,0,values=0,axis=0)
    first = (x.T.dot(sigmoid(x.dot(theta))-y))/len(x)

    return reg + first
def finall(theta,x,y,lamda,k): #优化
    for i in range(1,k+1):
        theta_i = np.zeros(x.shape[1],)
        res = minimize(fun=costfunction,x0=theta_i,args=(x,y==i,lamda),method='TNC',jac=gradient)
        theta[i-1,:] = res.x
    return theta
def predict(theta,x): #预测
    h = sigmoid(x.dot(theta.T))
    h_argmax = np.argmax(h,axis=1) #返回最大值索引
    return h_argmax+1 #真正标签
def plot_100_image(x):
    index = np.random.choice(len(x),100)#随机选择00
    images = x[index,:]
    fig,ax = plt.subplots(ncols = 10,nrows = 10 ,figsize=(10,10),sharex=True,sharey=True) #共享x，y轴
    for i in range(10):
        for j in range(10):
            ax[i,j].imshow(images[10*i+j,:].reshape(20,20).T,cmap='gray')
    plt.xticks([]) #为空不显示刻度
    plt.yticks([])
    plt.show()
data = loadmat('ex3data1.mat')  #加载数据,所以用scipy库 返回字典，X其实是图片
raw_x = data['X']
raw_y = data['y']


#print(raw_x.shape,raw_y.shape)
x = np.insert(raw_x,0,values=1,axis=1)
y = raw_y.flatten() #转换维度为了运算

theta = np.zeros((k,x.shape[1]))

theta_final = finall(theta,x, y, lamda, k)
#print(theta_final)
y_predict = predict(theta_final,x)

accurrity = np.mean(y_predict==y)

print(accurrity)
# plot_100_image(raw_x) #绘图