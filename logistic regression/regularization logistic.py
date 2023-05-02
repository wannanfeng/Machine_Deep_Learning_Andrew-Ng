import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import scipy.optimize as opt
#正则化 保留所有特征，减小参数
def get_y(df):  #读取标签值
    return np.array(df.iloc[:,-1])  #这样可以转换为数组
def sigmoid(z):
    return 1/(np.exp(-z)+1)

def feature_mapping(x,y,power): #特征映射
 # 当逻辑回归问题较复杂，原始特征不足以支持构建模型时，可以通过组合原始特征成为多项式，创建更多特征，使得决策边界呈现高阶函数的形状，从而适应复杂的分类问题。
    data={}
    for i in np.arange(power+1):
        for p in np.arange(i+1):
            data["f{}{}".format(i-p,p)] = np.power(x,i-p)*np.power(y,p)
    return pd.DataFrame(data)
def gradient(theta,X,y):    #实现求梯度
    return X.T.dot((sigmoid(X.dot(theta))-y))/len(X)

def regularized_gradient(theta,X,y,learningRate=1):
    theta_new = theta[1:]   #不加θ_0
    regularized_theta = (learningRate/len(X))*theta_new
    regularized_term = np.concatenate([np.array([0]),regularized_theta])    #前面加上0，是为了加给θ_0
    return gradient(theta,X,y)+regularized_term

def regularized_cost(theta, X, y, learningRate=1):  #实现代价函数
    first = np.multiply(-y, np.log(sigmoid(X.dot(theta.T))))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X.dot(theta.T))))
    reg = (learningRate/(2*len(X)))*np.sum(np.power(theta[1:theta.shape[0]],2))
    return np.sum(first - second) / (len(X)) + reg


path = 'ex2data2.txt'
data = pd.read_csv(path,header=None,names=['test1','test2','accepted'])
X = feature_mapping(np.array(data.test1),np.array(data.test2),power=6)

y = get_y(data)
theta = np.zeros(X.shape[1])    #返回numpy.ndarray类型

res = opt.minimize(fun=regularized_cost,x0=theta,args=(X,y),jac=regularized_gradient,method='Newton-CG')
theta_res = res.x
# print(res.x)
# print(regularized_gradient(theta,X,y))
# print(regularized_cost(theta,X,y))

fig, ax = plt.subplots(figsize=(12, 8))
positive = data[data['accepted']==1]
negative = data[data['accepted']==0]
ax.scatter(positive['test1'], positive['test2'], s=30, c='b', marker='o', label='Accepted')
ax.scatter(negative['test1'], negative['test2'], s=30, c='r', marker='x', label='Not Accepted')
# 添加图例
ax.legend()
ax.set_xlabel('test1')
ax.set_ylabel('test2')
plt.title("Regularized Logistic Regression")

#绘制决策边界
x = np.linspace(-1, 1.5, 250)
xx, yy = np.meshgrid(x, x)  #生成网格点坐标矩阵 https://blog.csdn.net/lllxxq141592654/article/details/81532855
z = feature_mapping(xx.ravel(), yy.ravel(), 6)  #xx,yy都是（250，250）矩阵，经过ravel后，全是(62500,1)矩阵。之后进行特征变换获取(62500,28)的矩阵
z = z.dot(theta_res)   #将每一行多项式与系数θ相乘，获取每一行的值=θ_0+θ_1*X_1+...+θ_n*X_n
z = np.array([z]).reshape(xx.shape) #将最终值转成250行和250列坐标
# print(z)
plt.contour(xx, yy, z, 0)   #传入的xx,yy需要同zz相同维度。
                            # 其实xx每一行都是相同的，yy每一列都是相同的，所以本质上来说xx是决定了横坐标，yy决定了纵坐标.
                            # 而z中的每一个坐标都是对于(x,y)的值---即轮廓高度。
                            # 第四个参数如果是int值，则是设置等高线条数为n+1---会自动选取最全面的等高线。　　如何选取等高线？具体原理还要考虑：是因为从所有数据中找到数据一样的点？
                            # https://blog.csdn.net/lanchunhui/article/details/70495353
                            # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contour.html#matplotlib.pyplot.contour
plt.show()