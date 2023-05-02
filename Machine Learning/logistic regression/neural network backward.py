import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
def sigmoid(z):
    return 1/(1+np.exp(-z))
def one_hot_encoder(raw_y): #进行one hot处理
    reslut = []
    for i in raw_y:
        y_temp = np.zeros(10)
        y_temp[i-1] = 1
        reslut.append(y_temp)
    return np.array(reslut) #list --> array
def fallten(a,b): #序列化权重参数
    return np.append(a.flatten(),b.flatten())
def unfallten(a):#解序列化权重参数
    theta1 = a[:25*401].reshape(25,401)
    theta2 = a[25*401:].reshape(10,26)
    return theta1,theta2
def forward(theta,x):#前向传播
    theta1,theta2 = unfallten(theta)
    a1 = x
    z2 = a1.dot(theta1.T)
    a2 = sigmoid(z2)
    a2 = np.insert(a2,0,values=1,axis=1)
    z3 = a2.dot(theta2.T)
    h = sigmoid(z3)
    return a1,z2,a2,z3,h
def costfuntion(theta,x,y):
    a1,z2,a2,z3,h = forward(theta,x)
    J = -np.sum(y*np.log(h)+(1-y)*np.log((1-h)))/len(x)
    return J
def reg_costfuntion(theta,x,y,lamda=1):

    theta1,theta2 = unfallten(theta)
    sum1 = np.sum(np.power(theta1[:,1:],2))
    sum2 = np.sum(np.power(theta2[:,1:],2))
    reg = (sum1+sum2)*lamda/len(x)/2
    return reg+costfuntion(theta,x,y)
def gradient_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))
def gradient(theta,x,y):
    theta1,theta2 = unfallten(theta)
    a1, z2, a2, z3, h = forward(theta, x)
    d3 = h - y
    d2 = d3.dot(theta2[:,1:])*gradient_sigmoid(z2)
    D2 = (d3.T.dot(a2))/len(x)
    D1 = (d2.T.dot(a1))/len(x)
    return fallten(D1,D2)
def reg_gradient(theta,x,y,lameda=1):
    D = gradient(theta,x,y)
    D1,D2 = unfallten(D)
    theta1,theta2 = unfallten(theta)
    D1[:,1:] = D1[:,1:] + theta1[:,1:]*lameda/len(x)
    D1[:,1:] = D1[:,1:] + theta1[:,1:]*lameda/len(x)
    return fallten(D1,D2)

def NN_training(x,y,lamda=10):
    init_theta = np.random.uniform(-0.5,0.5,10285)#均匀取-0.5到0.5中10285个
    res = minimize(fun=reg_costfuntion,x0=init_theta,args=(x,y,lamda),method='TNC',jac=reg_gradient,options={'maxiter':300}) #options最大迭代次数
    return res
def plt_hide_layer(theta):#绘制隐藏层
    theta1,theta2 = unfallten(theta)
    hidden_layer = theta1[:,1:]
    fig,ax = plt.subplots(ncols=5,nrows=5,figsize=(8,8),sharex=True,sharey=True)

    for i in range(5):
        for k in range(5):

            ax[i,k].imshow(hidden_layer[5*i+k].reshape(20,20).T,cmap = 'gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

data = sio.loadmat("ex4data1.mat")
theta0 = sio.loadmat('ex4weights.mat')
raw_x = data['X']
raw_y = data['y']
raw_y_1 = data['y'].reshape(5000,)
x = np.insert(raw_x,0,values=1,axis=1)
y = one_hot_encoder(raw_y)

theta1, theta2 = theta0['Theta1'], theta0['Theta2']
#print(theta1.shape,theta2.shape)#(25,401),(18,26)
theta = fallten(theta1,theta2)

# print(reg_costfuntion(theta,x,y))
res = NN_training(x,y)
a1,z2,a2,z3,h = forward(res.x,x)
y_predict = np.argmax(h,axis=1)+1
accurrtiy = np.mean(y_predict==raw_y_1)

print(accurrtiy)
plt_hide_layer(res.x)