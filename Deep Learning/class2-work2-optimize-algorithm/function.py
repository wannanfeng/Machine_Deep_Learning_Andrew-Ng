def original_gradient_descent(parameters,learning_rate,grads):
    L = len(parameters) // 2  # 神经网络的层数
    # 更新每个参数
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters
