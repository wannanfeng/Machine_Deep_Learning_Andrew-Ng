import numpy as np
import numpy as pd
import matplotlib.pyplot as plt
import pandas as pd

alph = 0.01
inters = 1000
def costfunction(x,y,theta):
    sum = np.power(x.dot(theta.T)-y,2)
    return np.sum(sum)/2/len(x)

def gradientdescent(x,y,theta,alph,inters):
    temp = np.matrix(np.zeros(theta.shape))
    k = int(theta.ravel().shape[1])
    cost = np.zeros(inters)
    for i in range(inters):
        cos =  x.dot(theta.T) - y
        for j in range(k):
            term = np.multiply(cos,x[:,j])
            temp[0,j] = theta[0,j] - alph*np.sum(term)/len(x)
        theta = temp
        cost[i] = costfunction(x,y,theta)
    return theta, cost


if __name__ == '__main__':
    data = pd.read_csv("ex1data2.txt",names=['size', 'bedroom', 'price'])
    #特征归一化 ： x =(x - u)/σ    u是均值   σ是标准差
    data = (data-data.mean())/data.std()
    data.insert(0,'ones',1)
    clos = data.shape[1] #几列
    x = data.iloc[:,0:clos-1]
    y = data.iloc[:,clos-1:clos]
    x = np.matrix(x.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array([0,0,0]))

    theta, cost = gradientdescent(x,y,theta,alph,inters)

    # 进行绘图 代价函数收敛图
    fig, ax = plt.subplots()  # 返回图表以及图表相关的区域，为空代表绘制区域为111--->一行一列图表，选中第一个
    ax.plot(np.arange(inters), cost, 'r') #arrange生成列表
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title("Error vs. Training Epoch")
    plt.show()