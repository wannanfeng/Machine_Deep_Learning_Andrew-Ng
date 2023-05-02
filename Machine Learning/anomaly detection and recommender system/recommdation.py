import sys
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')
def serilalize(x,theta): #序列化参数，因为后面调用scipy的优化算法时候需要传入一维数组
    return np.append(x.flatten(),theta.flatten())
#解序列化，传入序列化后的params
def de_serilalize(params,n_movie,n_user,n_feature):

    x = params[:n_movie*n_feature].reshape(n_movie,n_feature)
    theta = params[n_movie*n_feature:].reshape(n_user,n_feature)
    return x,theta
#协同过滤代价函数
def cost_function(params,y,n_movie,n_user,n_feature,r,lamda):
    x,theta = de_serilalize(params,n_movie,n_user,n_feature)
    J = np.sum(np.power((x.dot(theta.T) - y)*r,2))/2 # *r 剔除了没有评价的点
    reg = np.sum(theta*theta)*lamda/2 + np.sum(x*x)*lamda/2
    return J + reg

def grad_function(params,y,n_movie,n_user,n_feature,r,lamda):
    x,theta = de_serilalize(params,n_movie,n_user,n_feature)

    x_grad = ((x.dot(theta.T)-y)*r).dot(theta) + lamda*x
    theta_grad = ((x.dot(theta.T)-y)*r).dot(x) + lamda*theta

    grad = serilalize(x_grad,theta_grad)
    return grad
def init_new_user(n_movie):
    my_ratings = np.zeros((n_movie,1)) #存放对所有电影评分
    # 该用户评价了少数电影
    my_ratings[9] = 5
    my_ratings[66] = 5
    my_ratings[96] = 5
    my_ratings[121] = 4
    my_ratings[148] = 4
    my_ratings[285] = 3
    my_ratings[490] = 4
    my_ratings[599] = 4
    my_ratings[643] = 4
    my_ratings[958] = 5
    my_ratings[1117] = 3
    return my_ratings

def nomarlize(y,r): #均值归一化
    y_mean = (np.sum(y,axis=1))/(np.sum(r,axis=1))#计算为一维数组
    y_mean = y_mean.reshape(-1,1)
    y_norm = (y - y_mean)*r
    return y_norm,y_mean


data = loadmat('ex8_movies.mat')
params = loadmat('ex8_movieParams.mat') #随机初始化好的参数 包含num user943，num movie1682，num feature10
y = data['Y'] #用户对电影的评分0-5（没有评价也是显示0）
r = data['R'] #用户是否对某个电影评价，0没有评价，1为评价
# print(y.shape,r.shape) #均为(1682,943)
x = params['X'] #获取电影的特征矩阵 (1682, 10)
theta = params['Theta'] #获取用户的特征矩阵 (943, 10)
num_user = int(params['num_users']) #获取用户数943
num_movie = int(params['num_movies'])   #获取电影数1682
num_feature = int(params['num_features'])   #获取用户、电影特征数10


# #减小数据集用来更快的测试代价函数的正确性
# n_users = 4
# n_movies = 5
# n_features = 3
#
# x = x[0:n_movies,0:n_features]
# theta = theta[0:n_users,0:n_features]
# y = y[0:n_movies,0:n_users]
# r = r[0:n_movies,0:n_users]
# print(x.shape,y.shape,theta.shape)
# params = serilalize(x,theta)


# np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
# np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。

my_ratings = init_new_user(num_movie) #初始化一个新用户
y = np.c_[y,my_ratings] #加入总打分
r = np.c_[r,np.where(my_ratings!=0,1,0)] #当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回1，当条件不成立时where返回0
# print(y.shape) #增加了一列所以重新获取电影数目等
num_movie,num_user = y.shape

y_norm,y_mean = nomarlize(y,r)
#初始化特征和参数
x_random = np.random.random((num_movie,num_feature))
theta_random = np.random.random((num_user,num_feature))
param_s = serilalize(x_random,theta_random)

x,theta = de_serilalize(param_s,num_movie,num_user,num_feature)
print('外面运行就没有报错，进入minimize就报错---\n',x.reshape(num_movie,num_feature))

lamda = 5
# sys.exit()

res = minimize(fun=cost_function,
               x0=param_s,
               args=(y_norm,r,num_movie,num_user,num_feature,lamda),
               method='TNC',jac=grad_function,options={'maxiter':100}
               )
params_fit = res.x
fit_x,fit_theta = de_serilalize(params_fit,num_movie,num_user,num_feature)

y_pred = fit_x.dot(fit_theta.T) #所有的预测
y_pred = y_pred[:,-1] + y_mean.flatten()#选择出刚刚新用户的预测值,因为均值归一化了所以加回来

index = np.argsort(-y_pred)#按大到小

movies = []
with open('movie_ids.txt','read') as f:
    for line in f:
        tokens = line.strip().split(' ')
        movies.append(' '.join(tokens[1:]))

for i in range(10):
    print(index[i],movies[index[i]],y_pred[index[i]])