import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import scipy.optimize as opt
import sys

iters = 200000
alph = 0.004
def sigmoidfunction(z):
    return 1/(1+np.exp(-z))

def costfunction(theta,x,y):
    first = np.multiply(-y,np.log(sigmoidfunction(x.dot(theta.T))))
    last = np.multiply(1-y,np.log(1-sigmoidfunction(x.dot(theta.T))))
    return np.sum(first-last)/len(x)

def gradiant(theta,x,y): #重点理解 计算梯度
    return (x.T).dot(np.array([(sigmoidfunction(x.dot(theta.T))-y)]).T).flatten()/len(x)

def gradiantdescent(x,y,theta,iters,alph):
    cost = np.zeros(iters)
    temp = np.ones(len(theta))
    for i in range(iters):
        temp = temp - gradiant(x,y,theta)*alph
        theta = temp
        cost[i] = costfunction(x,y,theta)
    return theta,cost

def predict(x,theta):   #对数据进行预测
    return (sigmoidfunction(x*theta.T)>=0.5).astype(int)  #实现变量类型转换

if __name__ == "__main__":
    data = pd.read_csv("ex2data1.txt",names=['exam1','exam2','admit'])
    p1 = data[data['admit']==1] #获取为1的
    p0 = data[data['admit']==0]
    data.insert(0,'ones',1)
    cols = data.shape[1]
    x = np.array(data.iloc[:,0:cols-1].values)
    y = np.array(data.iloc[:,cols-1].values)
    theta = np.zeros(x.shape[1])
    #此方法求出结果不符合？？？ alph，iters小?
#    theta,cost = gradiantdescent(x,y,theta,iters,alph)

    # 使用高级优化算法 最小化函数
    res = opt.minimize(fun=costfunction, x0=theta, args=(x, y), jac=gradiant, method='TNC')
    #fun表示优化的函数，x0为初始猜测值（一维数组），args额外传递给优化函数的参数
    #method：求解的算法; jac:返回梯度向量的函数

   # print(res)
    theta_res = res.x  # 获取拟合的theta参数
   # print(theta_res)

#    y_pred = predict(x, theta_res)
#    print(classification_report(y, y_pred)) #模型评估函数，传入真实值和预测值，即预测中了几次
    





    # fig, ax = plt.subplots(figsize=(12, 8)) #绘制sigmoid图像
    # ax.plot(np.arange(-10, 10, step=0.01),sigmoidfunction(np.arange(-10, 10, step=0.01))) # 设置x,y轴各自数据，绘制图形
    # ax.set_xlabel('z', fontsize=18)
    # ax.set_ylabel('g(z)', fontsize=18)
    # ax.set_title('sigmoid function', fontsize=18)
    # plt.show()

    fig, ax = plt.subplots(figsize=(10, 5))  # 获取绘图对象
    ax.scatter(p1['exam1'], p1['exam2'], s=30, c='b', marker='o', label='Admitted')
    ax.scatter(p0['exam1'], p0['exam2'], s=30, c='r', marker='x', label='Not Admitted')
    # 添加图例
    ax.legend()
    ax.set_xlabel('Exam1 Score')
    ax.set_ylabel('Exam2 Score')
    plt.title("fitted curve vs sample")

    # 绘制决策边界
#    print('1 theta_res',theta_res)
    exam_x = np.arange(x[:, 1].min(), x[:, 1].max(), 0.01)
    theta_res = - theta_res / theta_res[2]  # 获取函数系数θ_0/θ_2 θ_0/θ_2
    print('2 theta_res',theta_res)
    exam_y = theta_res[0] + theta_res[1] * exam_x
    plt.plot(exam_x, exam_y)
    plt.show()