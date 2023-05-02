import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
iters = 1000 #迭代次数
alph = 0.01 #
def gradientdecent(x,y,theta,iters,alph):
    temp = np.matrix(np.zeros(theta.shape))
    suntheta = int(theta.ravel().shape[1]) #所需要求取的参数个数 reval()指把数组拉成一维度的

    for i in range(iters):
        cost1 = x.dot(theta.T)-y
        for j in range(suntheta):
            term = np.multiply(cost1,x[:,j])
            temp[0,j] = temp[0,j] - alph*np.sum(term)/len(x)
        theta = temp
        cost = costfunction(x,y,theta)
    return theta , cost


def costfunction(x,y,theta):
    k = np.power((x.dot(theta.T)-y),2) #power计算n次方
    return sum(k)/2/len(x)

if __name__ == '__main__':

    data1 = pd.read_csv("ex1data1.txt",names=['population','profit'])
    # print(data1.head())
    cols = data1.shape[1]    #获取列数   shape[0]是行数
    x = data1.iloc[:,0:cols-1]   #获取数据集
    y = data1.iloc[:,cols-1:cols]    #获取标签值---目标变量
    x.insert(0,'ones',1)
    # print(x)

    x = np.matrix(x.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0,0]))

    theta , cost = gradientdecent(x,y,theta,iters,alph) #回归结果
    # print(theta,cost)

    #进行绘图
    x = np.linspace(data1.population.min(),data1.population.max())    #抽取100个样本
    f = theta[0,0]+(theta[0,1]*x)   #线性函数，利用x抽取的等距样本绘制线性直线

    fig, ax = plt.subplots(figsize=(12,8))    #返回图表以及图表相关的区域，为空代表绘制区域为111--->一行一列图表，选中第一个
    ax.plot(x,f,'r',label="Prediction") #绘制直线
    ax.scatter(data1.population,data1.profit,label='Training Data')    #绘制散点图
    ax.legend(loc=4)    #显示标签位置　　给图加上图例　　'lower right'  : 4,
    ax.set_xlabel("Population")
    ax.set_ylabel("Profit")
    ax.set_title("Predicted Profit vs Population Size")
    plt.show()