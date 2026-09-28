import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from pprint import PrettyPrinter
from tabulate import tabulate
import math
from copy import deepcopy


###################################################################################################################################################
# -------------------------------
# 1. 设置中文字体和负号显示
# -------------------------------

# 检查可用字体（可选，调试用）
# print(matplotlib.rcParams['font.family'])

# 设置字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 尝试这些中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号 '-'

#Ex5.7根据式5.18和5.19，试构造一个能解决异或问题的RBF神经网络。

#初始化类变量
ini_v1=[0,0,1,1];ini_v2=[0,1,0,1]; fai_true=[0,1,1,0]            
v1=0;v2=0
beta1=1;beta2=1
c1=[0,1];c2=[1,0]
h1=h2=fai=0
w1=1;w2=1
eta=0.1
epoch = 1000
loss_sum=0

for itr in range(epoch):
    v1=ini_v1[itr%4];v2=ini_v2[itr%4]
    h1 = math.exp(-beta1*((v1-c1[0])**2+(v2-c1[1])**2))
    h2 = math.exp(-beta2*((v1-c2[0])**2+(v2-c2[1])**2))
    fai= w1*h1 +w2*h2
    loss=1/2*(fai_true[itr%4]-fai)**2
    #loss_sum+=loss
    print("v1:",v1,"  v2:",v2,"  h1:",h1,"  h2:",h2,"  fai:",fai,"  loss",loss)
    dloss_dfai= fai-fai_true[itr%4]
    dfai_dw1=h1;dfai_dw2=h2
    dfai_dbeta1=-w1*h1*((v1-c1[0])**2+(v2-c1[1])**2)
    dfai_dbeta2=-w2*h2*((v1-c2[0])**2+(v2-c2[1])**2)
    delta_w1=-eta*dloss_dfai*dfai_dw1;delta_w2=-eta*dloss_dfai*dfai_dw2
    delta_beta1=-eta*dloss_dfai*dfai_dbeta1;delta_beta2=-eta*dloss_dfai*dfai_dbeta2
    w1=w1+delta_w1;w2=w2+delta_w2
    beta1=beta1+delta_beta1;beta2=beta2+delta_beta2
    print("beta1:",beta1,"  beta2:",beta2,"  w1:",w1,"  w2:",w2)
print("\n")
v1,v2 = input("请输入异或门的两个输入(0或1),并用空格分隔  ->").split()
v1=int(v1);v2=int(v2)
h1 = math.exp(-beta1*((v1-c1[0])**2+(v2-c1[1])**2))
h2 = math.exp(-beta2*((v1-c2[0])**2+(v2-c2[1])**2))
fai= w1*h1 +w2*h2
if fai < 0.5:
    fai = 0
else:
    fai = 1
print(f"输入: [{v1}, {v2}] -> 预测: {fai%2} (真实: {(v1+v2)%2})")