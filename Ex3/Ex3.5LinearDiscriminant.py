#首次运行前需运行环境已安装相应的包pip install numpy pandas matplotlib scikit-learn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics

dataset = pd.read_csv(r'C:\Users\tanchao\pythonPrj\helloWorld\zhouEx\watermelon3a.csv',encoding='gbk')
#数据预处理
X = dataset[['密度','含糖率']]     #选择密度和含糖率两列
Y = dataset['好瓜']                 #选择好瓜这一列
good_melon = dataset[dataset['好瓜'] == 1]  #选择好瓜为1的行记录
bad_melon = dataset[dataset['好瓜'] == 0]
#画图
#f1 = plt.figure(1)              #创建空白图表
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.xlim(0,1)                   #设置x轴范围
plt.ylim(0,1)
plt.scatter(bad_melon['密度'],bad_melon['含糖率'],marker='x',color='r',s=50,label='bad')   #s表示描点的大小，设定横轴为密度，纵轴为含糖量
plt.scatter(good_melon['密度'],good_melon['含糖率'],marker='o',color='g',s=50,label='good')
plt.legend(loc='upper right')
#分割训练集和验证集，random_state=0确保每次运行代码时，分割方式都一样（可重复实验） 如果不设，每次运行会随机分，结果不稳定，测试集占总数据的比例0.25
X_train,X_test,Y_train,Y_test =(
model_selection.train_test_split(X,Y,test_size=0.25,random_state=0))
#创建线性分类模型
LDA_model = LinearDiscriminantAnalysis()
#训练模型
LDA_model.fit(X_train,Y_train)
#验证测试数据
Y_pred = LDA_model.predict(X_test)
#汇总
print(metrics.confusion_matrix(Y_test, Y_pred))  #统计结果判断正误，生成TP、FN、FP、TN统计数量的混淆矩阵
print(metrics.classification_report(Y_test, Y_pred))   #该报告展示了分类模型的查准率（precision）、查全率（recall）、F1分数（f1-score）以及各类别的样本数量（support）
print(LDA_model.coef_)                          #输出对率回归模型训练得到的“特征权重”（即模型学到的参数）

X_pred = np.linspace(0,1,100)                   #从0开始到1，按照(1-0)/99的间隔均匀分布，总共100个点
#根据模型参数生成曲线描点坐标，coef_ 是一个二维数组，形状为 (1, 2)，例如：[[5.88, 10.59]]，下述代码将分别取出模型学到的参数进行赋值
omega1, omega2 = LDA_model.coef_[0][0], LDA_model.coef_[0][1] 
b_intercept = LDA_model.intercept_
plt.plot(X_pred, -b_intercept/omega2-omega1/omega2*X_pred,color='k', linestyle='--')   #根据公式z=ω1*x1+ω2*x2+b，在令z=0，得到决策边界直线方程
plt.show()


