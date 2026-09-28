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


df_colors_cate = dataset.drop_duplicates()
grouped = dataset.groupby('色泽')
row_grouped = grouped['色泽'].count()
df_density = dataset[dataset['好瓜']==0]
print(dataset.columns)
lst_cols = [col for col in dataset.columns]         #提取列名，生成数据列名列表
lst_cols.pop(0), lst_cols.pop(-1)                       #去除首列和末列
label_cate = dataset[dataset.columns[-1]].unique()      #类别列表
dic_Bayes_featuresStat = {col:{} for col in lst_cols}    #转换为空字典(去除标签列)
mean = std = 0
int_sampleNum = len(dataset)                            #数据帧的记录数

int_Recrd = -1                                      #初始化查询记录序号
if int_Recrd <0 or int_Recrd >= int_sampleNum:
    int_Recrd = int((input('Please input the No. of the single chosen sample:(0~{n}) '.
                   format(n = int_sampleNum-1))).strip())
df_query = dataset.iloc[int_Recrd,:]


dict_pred = {}

for itr_label in label_cate:                        #遍历类别标签
     p_c_xi = p_c_xi_xj = 0
     for itr_colName1 in lst_cols:                        #遍历数据帧中所有数据列
         df_tmp1 = dataset[dataset[dataset.columns[-1]]==itr_label]   #筛选当前类别记录
         dtype_tmp1 = dataset.loc[int_Recrd,itr_colName1]               #获取当前属性值xi 
         df_tmp1 = df_tmp1[df_tmp1[itr_colName1]==dtype_tmp1]           #筛选当前属性值对应的记录
         if len(df_tmp1) != 0:                                        #当前记录非空
             p_c_xi += ((len(df_tmp1)+1)/(int_sampleNum+len(dataset[
                itr_colName1].unique())))                            #累加条件类别概率
             for itr_colName2 in lst_cols:                          #再次遍历数据列
                  dtype_tmp2 = dataset.loc[int_Recrd,itr_colName2] 
                  df_tmp2 = df_tmp1[df_tmp1[itr_colName2]==dtype_tmp2]
                  p_c_xi_xj = (len(df_tmp2) + 1)/(len(df_tmp1)+len(dataset[
                    itr_colName2].unique()))
                  p_c_xi *= p_c_xi_xj
     dict_pred[str(itr_label)]=p_c_xi
 
print(dict_pred)            


