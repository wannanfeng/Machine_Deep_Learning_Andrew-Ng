import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
def plot_image(x):
    # ax指生成小画面
    fig,ax = plt.subplots(nrows=8,ncols=8,figsize=(10,10),sharex=True,sharey=True) #整个窗口分成8行8列
    for i in range(8):
        for k in range(8):
            ax[i,k].imshow(x[8*i+k,:].reshape(32,32).T,cmap='gray') #取出前64张
    plt.xticks([])#为空不显示刻度
    plt.yticks([])
    plt.show()
data = loadmat('ex7faces.mat')
# print(data.keys())
x_before = data['X']
# print(x)
# print(x.shape) #(5000,1024)
K = 36
x = x_before - np.mean(x_before,axis=0)
C = x.T.dot(x)/len(x)
U,S,V = np.linalg.svd(C)
U1 = U[:,:K]
z_down = x.dot(U1) #降维的数据
z_up = z_down.dot(U1.T) #升维
plot_image(z_up)
plot_image(x_before)