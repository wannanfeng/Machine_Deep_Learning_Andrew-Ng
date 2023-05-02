import matplotlib.pyplot as plt
import numpy as np
from planar_utils import *
from testCases import *
import warnings
warnings.filterwarnings('ignore')
def testLogistic(X,Y): #测试逻辑回归的准确性
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T,Y.T)
    # y_pred = clf.predict(x)
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)  # 绘制决策边界
    plt.title("Logistic Regression")  # 图标题
    LR_predictions = clf.predict(X.T)  # 预测结果
    print("逻辑回归的准确性： %d " % float((np.dot(Y, LR_predictions) +
                                           np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
          "% " + "(正确标记的数据点所占的百分比)")

# 构建神经网络的一般方法是：
#
# 定义神经网络结构（输入单元的数量，隐藏单元的数量等）。
# 初始化模型的参数
# 循环：
    # 实施前向传播
    # 计算损失
    # 实现向后传播
    # 更新参数（梯度下降）
def layer_structure(x,y):#定义网络结构 one hiden layer
    n_x = x.shape[0]  # 输入层的节点数量
    n_h = 4  # 隐藏层的节点数量
    n_y = y.shape[0]  # 输出层的节点数量
    return n_x, n_h, n_y
def initalize_w_b(n_x,n_h,n_y):
    np.random.seed(2)
    w1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros(shape=(n_h,1))
    w2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros(shape=(n_y,1))

    # 使用断言确保数据格式是正确的
    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}

    return parameters
def forward(params,x):   # 前向传播
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']
    z1 = np.dot(w1, x) + b1
    A1 = np.tanh(z1)
    z2 = np.dot(w2, A1) + b2
    A2 = sigmoid(z2)  # 输出的为0-1
    assert (A2.shape == (1, x.shape[1]))
    cache = {"Z1": z1,
             "A1": A1,
             "Z2": z2,
             "A2": A2}
    return A2, cache
def costfuntion(A2,y):  # 代价函数
    m = y.shape[1]
    first = np.multiply(y, np.log(A2))
    second = np.multiply(1-y, np.log(1-A2))
    cost = (-1/m) * np.sum(first + second)

    cost = float(np.squeeze(cost))
    assert (isinstance(cost, float))
    return cost
def backforward(params,cache,x,y):  # 反向传播
    m = y.shape[1]
    w1 = params['w1']
    w2 = params['w2']
    A1 = cache['A1']
    A2 = cache['A2']
    dz2 = A2-y
    dw2 = (1/m) * np.dot(dz2, A1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(A1, 2))
    dw1 = (1/m) * np.dot(dz1, x.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
    grads = {'dw1':dw1,
             'dw2':dw2,
             'db1':db1,
             'db2':db2 }
    return grads
def update_w_b(grads,params,learningrate):
    w1, w2 = params['w1'], params['w2']
    b1, b2 = params['b1'], params['b2']
    w1 = w1 - learningrate * grads['dw1']
    b1 = b1 - learningrate * grads['db1']
    w2 = w2 - learningrate * grads['dw2']
    b2 = b2 - learningrate * grads['db2']
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}

    return parameters
def final_model(x,y,learningrate,iters):
    np.random.seed(3)  # 指定随机种子
    n_x, n_h, n_y = layer_structure(X, Y)
    params = initalize_w_b(n_x, n_h, n_y)
    costs = []
    for i in range(int(iters)):
        A2,cache = forward(params, x)
        cost = costfuntion(A2, y)
        grads = backforward(params, cache, x, y)
        params = update_w_b(grads, params, learningrate)
        if i % 1000 == 0:
            costs.append(cost)
    return params, costs

def predict(params,x):
    A2, cache = forward(params, x)
    y_pred = np.around(A2)
    return y_pred

if __name__ == '__main__':
    X, Y = load_planar_dataset() #训练集
    shape_X = X.shape  # (2, 400)
    shape_Y = Y.shape  # (1, 400)
    m = Y.shape[1]  # 训练集里面的数量
    learningrate, iters = 0.5, 5000
    # testLogistic(X,Y) #测试逻辑回归
    params, costs = final_model(X, Y, learningrate, iters)
    n_h = layer_structure(X, Y)[1]
    plot_decision_boundary(lambda x:predict(params, x.T), X, Y)
    plt.title("Hidden Layer of size: " + str(n_h))
    plt.show()
    y_pred = predict(params, X)
    print('learningrate: ' + str(learningrate) + ' iters: ' + str(iters))
    print('hiden layer: ' + str(n_h))
    print('准确率: %d' % float((np.dot(Y, y_pred.T) + np.dot(1 - Y, 1 - y_pred.T)) / float(Y.size) * 100) + '%')

    # print(costs)

    # plt.figure()
    # plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral) #绘制散点图
    #
    # plt.show()