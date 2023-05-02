from scipy.io import loadmat
from sklearn import svm
#想要知道如何将原始文件（emailSample1.txt）经过自然语言处理转为向量    请访问https://www.cnblogs.com/ssyfj/p/12931641.html
#此为导入经过自然语言处理对邮件进行向量化后的mat文件
#spamTrain.mat是对邮件进行预处理后（自然语言处理）获得的向量。
data1 = loadmat("spamTrain.mat")
x_train = data1["X"]
y_train = data1['y'].flatten()
# print(x_train,x_train.shape)
# x每行表示一封邮件,一封邮件有1899个特征由0，1表示。0为当前邮件没有语意库里包含的单词，1表示可以在语意库找到单词
data2 = loadmat('spamTest.mat')
x_test = data2['Xtest']
y_test = data2['ytest'].flatten()
Cvalues = [3,10,30,100,0.01,0.03,0.1,0.3,1]
best_score = 0
best_param = 0
for c in Cvalues:
    svc1 = svm.SVC(C=c,kernel='linear')
    svc1.fit(x_train,y_train)
    scroe = svc1.score(x_test,y_test)
    if scroe>best_score:
        best_score = scroe
        best_param = c
print(best_score,best_param)
best_svc = svm.SVC(C=best_param,kernel='linear')
best_svc.fit(x_train,y_train)
score_train = best_svc.score(x_train,y_train)
score_test = best_svc.score(x_test,y_test)
print(score_train,score_test)
