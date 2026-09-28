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

dataset = pd.read_csv(r'D:\vs_prj\python\MLPrj\pythonPrj2026\meloneBuch\watermelon3.0c.csv',encoding='gbk')
print(dataset)
#print(dataset.columns)
#df_colors_cate = dataset['色泽'].unique()
df_colors_cate = dataset.drop_duplicates()
#print(df_colors_cate)
grouped = dataset.groupby('色泽')
#print(grouped['色泽'].count())
row_grouped = grouped['色泽'].count()
#print(row_grouped.shape)
df_density = dataset[dataset['好瓜']==0]
#print(df_density['密度'].std())
print(dataset.columns)
lst_cols = [col for col in dataset.columns]         #提取列名，生成数据列名列表
lst_cols.pop(0), lst_cols.pop(-1)                                 #去除首列
label_cate = dataset[dataset.columns[-1]].unique()      #类别列表
dic_Bayes_featuresStat = {col:{} for col in lst_cols}    #转换为空字典(去除标签列)
mean = std = 0
int_sampleNum = len(dataset)                            #数据帧的记录数

for itr_colName in lst_cols:                           #遍历数据帧中所有数据列
    for itr_label in label_cate:                        #遍历类别标签
        if dataset.loc[:, itr_colName].dtype == 'str':                                      #判断列是否存储字符串数据
            df_groupedLabelSet = dataset[dataset[dataset.columns[-1]] == itr_label]         #筛选当前标签类的子集 
            df_groupedLabelSet = df_groupedLabelSet.groupby(itr_colName).count()            #分类汇总当前类下不同属性值

            dic_Bayes_featuresStat[itr_colName][str(itr_label)] = dict([[indexGp,featuresGp] #汇总后的列表转换为字典项
                for indexGp,featuresGp in df_groupedLabelSet.iloc[:,0].items()])
        else:                                                                                #实数数据列求均值和样本标准差
            df_groupedLabelSet = dataset[dataset[dataset.columns[-1]] == itr_label]
            mean = df_groupedLabelSet[itr_colName].mean()
            std = df_groupedLabelSet[itr_colName].std()
            dic_Bayes_featuresStat[itr_colName][str(itr_label)] = {'mean':float(mean), 'std':float(std)}

            

                        
print(dic_Bayes_featuresStat)

int_Recrd = -1                                      #初始化查询记录序号
if int_Recrd <0 or int_Recrd >= int_sampleNum:
    int_Recrd = int((input('Please input the No. of the single chosen sample:(0~{n}) '.
                   format(n = int_sampleNum-1))).strip())
df_query = dataset.iloc[int_Recrd,:]
str_color = df_query.iloc[1]
str_root = df_query.iloc[2]
str_knock = df_query.iloc[3]
str_pattern = df_query.iloc[4]
str_navel = df_query.iloc[5]
str_touch = df_query.iloc[6]
real_density = df_query.iloc[7]
real_sugar = df_query.iloc[8]
#计算拉普拉斯平滑后的贝叶斯条件分类属性概率
int_label0 = len(dataset[dataset[dataset.columns[-1]]==0])      #0类记录数           
int_label1 = len(dataset[dataset[dataset.columns[-1]]==1])      #1类记录数
p_label0 = (int_label0+1)/(int_label0+int_label1+len(label_cate))   #0类占比(概率)
p_label1 = (int_label1+1)/(int_label0+int_label1+len(label_cate))   #1类占比(概率)
int_featureNum_color = len(dataset['色泽'].unique())         #首列属性的数值种类数
p_color0 = (dic_Bayes_featuresStat['色泽']['0'].get(str_color,0)+1
            )/(int_label0+int_featureNum_color)                           #待查询首个属性对应的0分类条件概率
p_color1 = (dic_Bayes_featuresStat['色泽']['1'].get(str_color,0)+1
            )/(int_label1+int_featureNum_color)                           #待查询首个属性对应的1分类条件概率

int_featureNum_root = len(dataset['根蒂'].unique())    
p_root0 = (dic_Bayes_featuresStat['根蒂']['0'].get(str_root,0)+1
            )/(int_label0+int_featureNum_root)
p_root1 = (dic_Bayes_featuresStat['根蒂']['1'].get(str_root,0)+1
            )/(int_label1+int_featureNum_root)

int_featureNum_knock = len(dataset['敲声'].unique())    
p_knock0 = (dic_Bayes_featuresStat['敲声']['0'].get(str_knock,0)+1
            )/(int_label0+int_featureNum_knock)
p_knock1 = (dic_Bayes_featuresStat['敲声']['1'].get(str_knock,0)+1
            )/(int_label1+int_featureNum_knock)

int_featureNum_pattern = len(dataset['纹理'].unique())    
p_pattern0 = (dic_Bayes_featuresStat['纹理']['0'].get(str_pattern,0)+1
            )/(int_label0+int_featureNum_pattern)
p_pattern1 = (dic_Bayes_featuresStat['纹理']['1'].get(str_pattern,0)+1
            )/(int_label1+int_featureNum_pattern)

int_featureNum_navel = len(dataset['脐部'].unique())    
p_navel0 = (dic_Bayes_featuresStat['脐部']['0'].get(str_navel,0)+1
            )/(int_label0+int_featureNum_navel)
p_navel1 = (dic_Bayes_featuresStat['脐部']['1'].get(str_navel,0)+1
            )/(int_label1+int_featureNum_navel)

int_featureNum_touch = len(dataset['触感'].unique())    
p_touch0 = (dic_Bayes_featuresStat['触感']['0'].get(str_touch,0)+1
            )/(int_label0+int_featureNum_touch)
p_touch1 = (dic_Bayes_featuresStat['触感']['1'].get(str_touch,0)+1
            )/(int_label1+int_featureNum_touch)

#计算属性为实数值的概率密度函数
p_density0 = math.exp(-(df_query['密度']-dic_Bayes_featuresStat['密度']['0']['mean']
                      )**2/2/(dic_Bayes_featuresStat['密度']['0']['std'])
                      **2)/((2*math.pi)**0.5)/dic_Bayes_featuresStat['密度']['0']['std']
p_density1 = math.exp(-(df_query['密度']-dic_Bayes_featuresStat['密度']['1']['mean']
                      )**2/2/(dic_Bayes_featuresStat['密度']['1']['std'])
                      **2)/((2*math.pi)**0.5)/dic_Bayes_featuresStat['密度']['1']['std']

#计算属性为实数值的概率密度函数
p_sugar0 = math.exp(-(df_query['含糖率']-dic_Bayes_featuresStat['含糖率']['0']['mean']
                      )**2/2/(dic_Bayes_featuresStat['含糖率']['0']['std'])
                      **2)/((2*math.pi)**0.5)/dic_Bayes_featuresStat['含糖率']['0']['std']
p_sugar1 = math.exp(-(df_query['含糖率']-dic_Bayes_featuresStat['含糖率']['1']['mean']
                      )**2/2/(dic_Bayes_featuresStat['含糖率']['1']['std'])
                      **2)/((2*math.pi)**0.5)/dic_Bayes_featuresStat['含糖率']['1']['std']
#计算用例为0类的概率
p_predict0 = math.log(p_label0*p_color0*p_root0*p_knock0*p_pattern0*p_navel0*p_touch0*p_density0*p_sugar0)

#计算用例为1类的概率
p_predict1 = math.log(p_label1*p_color1*p_root1*p_knock1*p_pattern1*p_navel1*p_touch1*p_density1*p_sugar1)

print(f'计算用例为0类的概率: {p_predict0:.10f}')
print(f'计算用例为1类的概率: {p_predict1:.10f}')
if p_predict0 > p_predict1:
    print(f'预测结果为 0')
else:
    print(f'预测结果为 1')