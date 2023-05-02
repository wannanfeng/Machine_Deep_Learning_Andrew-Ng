import numpy as np
from dnn_utils import *
import matplotlib.pyplot as plt
def inital_w_b(layers_dim):   #  初始化多层

    # layers_dims - 包含我们网络中每个图层的节点数量的列表
    np.random.seed(3)
    params = {}
    for i in range(1,len(layers_dim)):
        params['w' + str(i)] = np.random.randn(layers_dim[i], layers_dim[i-1]) / np.sqrt(layers_dim[i-1])
        params['b' + str(i)] = np.zeros(shape=(layers_dim[i], 1))

        # 确保我要的数据的格式是正确的
        assert (params["w" + str(i)].shape == (layers_dim[i], layers_dim[i - 1]))
        assert (params["b" + str(i)].shape == (layers_dim[i], 1))

    return params
def costfuntion(AL,y):
    m = y.shape[1]
    cost = (-1/m) * np.sum(np.multiply(y, np.log(AL)) + np.multiply(1-y, np.log(1-AL)))

    cost = np.squeeze(cost)
    assert (cost.shape == ())
    return cost
def linear_forword(A,w,b):  # 线性求和
    z = np.dot(w,A) + b
    assert (z.shape == (w.shape[0], A.shape[1]))
    cache = (A, w, b)
    return z, cache

def activate_forward(A_prev,w,b,activate):   # 前向传播激活值计算
    # A_prev 为上层的激活值
    if activate == 'sigmoid':
        z, linear_cache = linear_forword(A_prev, w, b)
        A, activate_cache = sigmoid(z)
    elif activate == 'relu':
        z, linear_cache = linear_forword(A_prev, w, b)
        A, activate_cache = relu(z)
    cache = (linear_cache, activate_cache)
    return A, cache

def muti_model_forward(x,params):  # 多层前向传播
    A = x
    caches = []
    l = len(params) // 2  # 地板除，只保留整数
    for i in range(1,l):
        A_prev = A
        A, cache = activate_forward(A_prev, params['w' + str(i)], params['b' + str(i)],'relu')
        caches.append(cache)
    AL, cache = activate_forward(A, params['w' + str(l)], params['b' + str(l)], 'sigmoid')
    caches.append(cache)
    assert (AL.shape == (1, x.shape[1]))
    return AL, caches

def backforward(dz,cache): # 反向传播梯度计算
    A_prev, w, b = cache
    m = A_prev.shape[1]
    dw = (1/m) * np.dot(dz, A_prev.T)
    db = (1/m) * np.sum(dz, axis=1, keepdims=True)
    dA_prev = np.dot(w.T, dz)

    return dA_prev, dw, db

def activate_backward(dA,cache,activate="relu"):
    linear_cache, activate_cache = cache
    if activate == 'relu':
        dz = relu_backward(dA, activate_cache)
        dA_prev, dw, db = backforward(dz, linear_cache)
    elif activate == 'sigmoid':
        dz = sigmoid_backward(dA, activate_cache)
        dA_prev, dw, db = backforward(dz, linear_cache)
    return dA_prev, dw, db

def muti_model_backforward(AL,Y,caches): #多层反向传播
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dw" + str(L)], grads["db" + str(L)] = activate_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activate_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dw" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_param(params, grads, learningrate):
    #更新参数
    l = len(params)//2
    for i in range(l):
        params['w'+str(i+1)] = params['w'+str(i+1)] - learningrate * grads['dw'+str(i+1)]
        params['b'+str(i+1)] = params['b'+str(i+1)] - learningrate * grads['db'+str(i+1)]
    return params


def muti_layer_model(X,Y,layers_dims,learning_rate = 0.0075,num_iterations = 3000,print_cost=False,isPlot=True):
    np.random.seed(1)
    costs = []

    parameters = inital_w_b(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = muti_model_forward(X, parameters)

        cost = costfuntion(AL, Y)

        grads = muti_model_backforward(AL, Y, caches)

        parameters = update_param(parameters, grads, learning_rate)

        # 打印成本值，如果print_cost=False则忽略
        if i % 100 == 0:
            # 记录成本
            costs.append(cost)
            # 是否打印成本值
            if print_cost:
                print("第", i, "次迭代，成本值为：", np.squeeze(cost))
    # 迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters


def predict(X, y, parameters):  #预测结果
    m = X.shape[1]
    n = len(parameters) // 2  # 神经网络的层数
    p = np.zeros((1, m))

    # 根据参数前向传播
    probas, caches = muti_model_forward(X, parameters)
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("准确度为: " + str(float(np.sum((p == y)) / m)))
    return p


def print_mislabeled_images(classes, X, y, p):
    #绘制预测和实际不同的图像。
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title(
            "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode(
                "utf-8"))





