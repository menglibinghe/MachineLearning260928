#首次运行前需运行环境已安装相应的包pip install numpy pandas matplotlib scikit-learn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows',None)
pd.set_option('max_colwidth',200)
pd.set_option('expand_frame_repr', False)
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

dataset = pd.read_csv(r'C:\Users\tanchao\Desktop\breast+cancer+coimbra\Breast_Cancer_Coimbra.csv',encoding='gbk')
#数据预处理
X = dataset.iloc[:,:-1]               #选择除了最后列的其他列
Y = dataset.iloc[:,-1]                #选择‘Classification’这一列
patientRec = dataset[dataset['Classification'] == 0]  #选择是病人的行记录
healthyRec = dataset[dataset['Classification'] == 1]
#画图
#f1 = plt.figure(1)              #创建空白图表

#分割训练集和验证集，random_state=0确保每次运行代码时，分割方式都一样（可重复实验） 如果不设，每次运行会随机分，结果不稳定，测试集占总数据的比例0.5
X_train,X_test,Y_train,Y_test =(
model_selection.train_test_split(X,Y,test_size=0.5,random_state=0))
#创建模型
log_model = LogisticRegression()
#训练模型
log_model.fit(X_train,Y_train)
#验证测试数据
Y_pred = log_model.predict(X_test)
#汇总输出
print(metrics.confusion_matrix(Y_test, Y_pred))  #统计结果判断正误，生成TP、FN、FP、TN统计数量的混淆矩阵
#输出查准率、查全率，会根据正例选择不同，所以有两行不同的读数
print(metrics.classification_report(Y_test, Y_pred))   #该报告展示了分类模型的查准率（precision）、查全率（recall）、F1分数（f1-score）以及各类别的样本数量（support）
print(log_model.coef_)                          #输出回归模型训练得到的“特征权重”（即模型学到的参数）
#coef_ 是一个二维数组，形状为 (1, 2)，例如：[[5.88, 10.59]]，下述代码将分别取出模型学到的参数进行赋值
