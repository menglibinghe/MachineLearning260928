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
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold,KFold         #StratifiedKFold 会自动按类别比例划分，使每个 fold 的类别分布与总体一致！

dataset = pd.read_csv(r'C:\Users\tanchao\Desktop\breast+cancer+coimbra\Breast_Cancer_Coimbra.csv',encoding='gbk')
#数据预处理
X = dataset.drop(columns=['Classification'])            #选择除了最后列的其他列
Y = dataset['Classification']                           #选择‘Classification’这一列
patientRec = dataset[dataset['Classification'] == 0]    #选择是病人的行记录
healthyRec = dataset[dataset['Classification'] == 1]
#画图
#f1 = plt.figure(1)              #创建空白图表


#十次十折交叉验证
def tenfolds():
    model = LogisticRegression(max_iter=1000)                                       #max_iter默认为100，这种情况下迭代会超限                
    truthLst = []                                                                    #统计准确率
    for run in range(10):                                                             #按照10次进行循环
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=run)   #shuffle表示打乱顺序后再划分,表示一次划分;random_state表示打乱用的随机种子数，控制打乱是否可以重现。run自动取值0，1，2，...
        for train_idx, test_idx in kf.split(X,Y):                                     #针对其中一次划分,遍历10折,轮流设定其中一折测试集(剩余子集为训练集)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]                   #存放划分好的记录索引号
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
            # 训练模型
            model.fit(X_train, Y_train)
            # 预测
            Y_pred = model.predict(X_test)
            accuracy = model.score(X_test, Y_test)                  # 直接获取本次 fold 的准确率，即以一折作为测试集,其他为训练集时准确率
            truthLst.append(accuracy)
    print("模型的十次十折交叉验证平均准确率是: ",sum(truthLst)/len(truthLst))
    print("模型的系数是: ",[f"{x:6f}" for x in model.coef_[0].tolist()])
tenfolds()
#################################################################################

#留一法
def leaveone():
    model = LogisticRegression(max_iter=1000)                               #max_iter默认为100，这种情况下迭代会超限
    lvn = model_selection.LeaveOneOut()
    truthLst = []
    for train_idx, test_idx in lvn.split(X,Y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]                   #存放划分好的记录索引号
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

        # 用回归进行训练，拟合数据
        model.fit(X_train, Y_train)
        # 用训练好的模型预测
        Y_pred = model.predict(X_test)
        if Y_pred == Y_test.item():                                           #因为Y_test存放变量是 pandas.Series 类型，需要通过.item()提取单个元素值  
            truthLst.append(1)
        else:
            truthLst.append(0)
    print("模型的留一法交叉验证平均准确率是: ",sum(truthLst)/len(truthLst))
    print("模型的系数是: ",[f"{x:6f}" for x in model.coef_[0].tolist()])    
    
leaveone()
