import sys

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
def select_epsilon(yval,p):
    best_epsilon = 0
    best_score = 0
    epsilon = np.linspace(min(p),max(p),1000) #在p最大和最小间均匀选择1k个候选值
    for e in epsilon:
        p_ = p < e #概率和epsilon对比，如果小于则为异常点  返回1
        tp = np.sum((yval==1)&(p_==1)) #真实值yval为1，预测也为1
        fp = np.sum((yval==0)&(p_==1))
        fn = np.sum((yval==1)&(p_==0))
        # 计算查准率和查全率
        if tp + fp == 0 or tp + fn == 0: #分母不为0
            continue
        prec = tp / (tp + fp)  # 查准率
        rec = tp / (tp + fn)  # 查全率
        score = 2*prec*rec/(prec+rec) if(prec+rec) else 0 #如果不为0执行前面，为0则赋为0
        if score>best_score:
            best_epsilon = e
            best_score = score
        return best_epsilon,best_score
def Parameter_gussian(x,iscovariance): #后者判断是否求解协方差还是正常方差
    u = np.mean(x,axis=0)
    if iscovariance: #如果为真则执行
        sigma = (x-u).T.dot(x-u)/len(x)
    else:
        k = np.power((x-u),2)
        sigma = np.sum(k,axis=0)/len(x)
    return u,sigma

def gaussian_distribution(x,u,sigma):
    if np.ndim(sigma)==1: #判断矩阵维度  如果获取方差则为一维矩阵，如果协方差则为高维
        sigma = np.diag(sigma) #转为对角线矩阵，便可用协方差公式计算
    x = x - u
    n = x.shape[1]
    first = np.power(2*np.pi,-n/2)*(np.linalg.det(sigma)**(-0.5))
    second = np.diag(x.dot(np.linalg.inv(sigma)).dot(x.T))
    p = first*np.exp(-second*0.5)
    # p = p.reshape(-1,1) #转为2维
    # p = (1/np.sqrt(2*np.pi*sigma))*np.exp(-np.power(x-u,2)/2/sigma) #此为方差求解方式
    # p = np.prod(p, axis=1)  # 横向累乘
    return p
def get_anomaly(x,p,best_epslion):
    anomaly = x[np.where(p<best_epslion)[0]]
    return anomaly
data = loadmat('ex8data2.mat')
x = data['X']
x_val = data['Xval']
y_val = data['yval'].flatten()

u,sigma = Parameter_gussian(x,iscovariance=False)

p = gaussian_distribution(x,u,sigma)
p_val = gaussian_distribution(x_val,u,sigma)
# print('p_val',p_val)
# print(data.keys())
best_epsilon,best_score = select_epsilon(y_val,p_val)
print(best_epsilon,best_score)
anomaly = get_anomaly(x,p,best_epsilon)
print(anomaly,len(anomaly))
