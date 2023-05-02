import sys

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def plt_origin():
    plt.scatter(x[:,0],x[:,1])
    plt.show()
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
def visualize_countours(u,sigma): #绘制高斯分布等高线
    # 由plt_origin可以知道，我们选取5-25，范围比较合适

    x = np.linspace(5, 25, 100)
    y = np.linspace(5, 25, 100)
    xx,yy = np.meshgrid(x,y)
    X = np.c_[xx.flatten(),yy.flatten()]  #数据对应网格中每一个点
    z = gaussian_distribution(X, u, sigma).reshape(xx.shape)  # 获取每一个点坐标的高斯值
    cont_levels = [10 ** h for h in range(-20, 0, 3)]  # 当z为当前列表的值才绘出等高线（最高1）   不设置的话，会比较奇怪
    plt.contour(xx, yy, z, cont_levels)
    plt_origin()
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
def get_and_plt_anomalys():
    # print(np.where(p<best_epsilon))
    anomalys = x[np.where(p < best_epsilon)[0]]
    # s为点的大小,facecolor为none即时设置空心点(透明点),edgecolors为边界线颜色, 这样可以实现圈出点的作用
    plt.scatter(anomalys[:,0],anomalys[:,1],s=80,facecolors='none',edgecolors='r')

data = loadmat('ex8data1.mat')
x = data['X']

x_val = data['Xval']
y_val = data['yval'].flatten()
u,sigma = Parameter_gussian(x,iscovariance=False) #后者判断是否求解协方差还是正常方差,ture则求解协方差
# print(u,sigma)
p = gaussian_distribution(x,u,sigma)
p_val = gaussian_distribution(x_val,u,sigma)

best_epsilon,best_score = select_epsilon(y_val,p_val)
print(best_score,best_epsilon)

get_and_plt_anomalys()
visualize_countours(u,sigma)

# print(data.keys())
# plt_origin()
