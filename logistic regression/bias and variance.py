from scipy.io import loadmat
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plotdata():
    fig,ax = plt.subplots()
    ax.scatter(x_train[:,1],y_train)
    ax.set(xlabel='x',ylabel='y')

def feature_mapping(x,power=6): #特征映射
 # 当逻辑回归问题较复杂，原始特征不足以支持构建模型时，可以通过组合原始特征成为多项式，创建更多特征，使得决策边界呈现高阶函数的形状，从而适应复杂的分类问题。
    for i in range(2,power+1):
        #这里为增加x的多次幂
        x = np.insert(x,x.shape[1],np.power(x[:,1],i),axis=1) #第i列的数进行i次幂
        return x
def get_means_std(x):#获取均值和方差，进行归一化,因为x平方了几次可能导致数据集变化大
    means = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    return means,std
def feature_nomalize(x,means,std):#归一化
    x[:,1:] = (x[:,1:]-means[1:])/std[1:]
    return x
def sigmoid(z):
    return 1/(1+np.exp(-z))
def reg_constfuntion(theta,x,y,lamda):
    cost = np.sum(np.power(x.dot(theta)-y.flatten(),2))
    reg = theta[1:].dot(theta[1:])*lamda
    return (cost+reg)/(2*len(x))
def reg_gradient(theta,x,y,lamda):
    grad = (x.dot(theta)-y.flatten()).dot(x)
    reg = lamda * theta
    reg[0] = 0 #剔除theta[0]
    return (grad+reg)/len(x)
def train_model(x,y,lamda):
    theta = np.ones(x.shape[1])
    res = minimize(fun=reg_constfuntion,x0=theta,args=(x,y,lamda),method='TNC',jac=reg_gradient)
    return res.x
#训练集和验证集数目改变时候两种损失函数的变化
def learning_curve(x_train,y_train,x_val,y_val,lamda):
    x = range(1,len(x_train)+1) #训练集个数
    train_cost = [] #存放训练集损失函数
    val_cost = [] #存放验证集损失函数
    for i in x:
        res = train_model(x_train[:i,],y_train[:i,],lamda)
        train_cost_i = reg_constfuntion(res,x_train[:i,],y_train[:i,],lamda) #1->i个
        val_cost_i = reg_constfuntion(res,x_val,y_val,lamda)
        val_cost.append(val_cost_i)
        train_cost.append(train_cost_i)
    plt.plot(x,train_cost,label = 'training cost')
    plt.plot(x,val_cost,label = 'val cost')
    plt.legend()
    plt.xlabel("number of train examples")
    plt.ylabel("error")
    plt.show()
def plt_feature_mapping():
    plotdata()
    x = np.linspace(-60,60,100)#绘制拟合函数
    xx = x.reshape(100,1)
    xx = np.insert(xx,0,1,axis=1)
    xx = feature_mapping(xx)
    xx = feature_nomalize(xx,train_means,train_std)
    plt.plot(x,xx.dot(train_final_two),c='r',linestyle='--')
    plt.show()

data = loadmat('ex5data1.mat')
# print(data.keys())
x_train,y_train = data['X'], data['y'] #训练集
x_val,y_val = data['Xval'],data['yval'] #验证集
x_test,y_test = data['Xtest'], data['ytest']#测试集
# print(x_train.shape,x_val.shape,x_test.shape)
x_train = np.insert(x_train,0,1,axis=1)
x_val = np.insert(x_val,0,1,axis=1)
x_test = np.insert(x_test,0,1,axis=1)
theta = np.ones(x_train.shape[1])
lamda = 1

theta_final = train_model(x_train,y_train,lamda)
# plotdata()
# plt.plot(x_train[:,1],x_train.dot(theta_final),c='r') #线性回归去拟合
# plt.show()
# learning_curve(x_train,y_train,x_val,y_val,lamda)

x_train_feature_mapping = feature_mapping(x_train) #特征映射
x_test_feature_mapping = feature_mapping(x_test)
x_val_feature_mapping = feature_mapping(x_val)
train_means,train_std = get_means_std(x_train_feature_mapping) #获取方差均值
x_train_normalize = feature_nomalize(x_train_feature_mapping,train_means,train_std) #归一化
x_text_normalize = feature_nomalize(x_test_feature_mapping,train_means,train_std)
x_val_normalize = feature_nomalize(x_val_feature_mapping,train_means,train_std)
train_final_two = train_model(x_train_normalize,y_train,lamda) #训练

plt_feature_mapping()
learning_curve(x_train_feature_mapping,y_train,x_val_feature_mapping,y_val,lamda)