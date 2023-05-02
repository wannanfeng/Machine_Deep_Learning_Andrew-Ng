import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset
import sys
# 建立神经网络的主要步骤是：
#
# 定义模型结构（例如输入特征的数量）
#
# 初始化模型的参数
#
# 循环：
    # 3.1 计算当前损失（正向传播）
    # 3.2 计算当前梯度（反向传播）
    # 3.3 更新参数（梯度下降）
def plt_inital_show():  # 初始化简单查看图片
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()  # 加载数据
    plt.figure()
    plt.imshow(train_set_x_orig[0])
    plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

def initialize_w_b(dim): #初始化w，b
    # w = np.random.randn(dim,1)
    w = np.zeros(shape=(dim,1))
    b = 0
    # 使用断言来确保我要的数据是正确的
    assert (w.shape == (dim, 1))  # w的维度是(dim,1)
    assert (isinstance(b, float) or isinstance(b, int))  # b的类型是float或者是int

    return w,b
def costfuntion(x,y,w,b): #代价函数
    A = np.dot(w.T,x)+b
    A = sigmoid(A)
    m = x.shape[1]
    cost = (-1/m)*np.sum(y*np.log(A)+(1-y)*(np.log(1-A)))
    return cost
def gradient(x,y,w,b): #更新梯度dw，db
    A = np.dot(w.T, x)+b
    A = sigmoid(A)
    m = x.shape[1]
    dw = (1/m)*np.dot(x,(A-y).T)
    assert (dw.shape == (w.shape[0], 1))
    db = (1/m)*np.sum(A-y)

    grad = {  # 存储为字典
            'dw': dw,
            'db': db
            }
    return grad
def optimize(x,y,w,b,alpha,iters):  # 更新w和d
    costs = []
    iterssum = []
    for i in range(int(iters)):
        grad = gradient(x,y,w,b)
        cost = costfuntion(x,y,w,b)
        w = w - alpha * grad['dw']
        b = b - alpha * grad['db']
        if i % 100 == 0: #每一百记录一个
            costs.append(cost)
            iterssum.append(i)
    params = {'w':w, 'b':b}
    return params, costs, iterssum

def y_predict(params ,x): #预测test_x
    m = x.shape[1] #图片数目
    w = params['w']
    b = params['b']
    y_pred = np.zeros((1,m))

    A = sigmoid(np.dot(w.T,x)+b)
    for i in range(m):
        if A[0,i] > 0.5:
            y_pred[0,i] = 1
        else:
            y_pred[0,i] = 0

    assert (y_pred.shape == (1,m))
    return y_pred
def mainmodel(x_train,x_test,y_train,y_test,alpha,iters):
    w, b = initialize_w_b(x_train.shape[0])
    params, costs,iterssum = optimize(x_train,y_train,w,b,alpha,iters)
    y_pred_test = y_predict(params,x_test) #预测测试集
    y_pred_train = y_predict(params,x_train) #预测训练集
    print('学习率为：',alpha,'迭代次数为：',iters)
    print("训练集准确性：", format(100 - np.mean(np.abs(y_pred_train - y_train)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(y_pred_test - y_test)) * 100), "%")

    return params,costs, iterssum
def init_data():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()  # 加载数据

    train_set_x_orig_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    # print(train_set_x_orig_flatten.shape) # 每张图片压缩为一位的数据, dim = (64*64*3, 209)
    test_set_x_orig_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

    # RGB全为0-255的数据，可以全部降为0-1居中化
    train_x = train_set_x_orig_flatten / 255
    test_x = test_set_x_orig_flatten / 255

    alpha, iters = 0.005, 2000 # 超参数:学习率和迭代次数

    return train_x,test_x,train_set_y,test_set_y,alpha,iters
def plot_cost():
    plt.figure()
    plt.plot(iterssum, costs, c='b', marker='1', linestyle='-')
    plt.xlabel('iters')
    plt.ylabel('costs')
    plt.title('learning rate = {}'.format(alpha))
    plt.show()
if __name__ == '__main__':
    x_train,x_test,y_train,y_test,alpha,iters = init_data()

    params, costs,iterssum = mainmodel(x_train,x_test,y_train,y_test,alpha,iters)
    # print('P',params,'C',costs)
    plot_cost()