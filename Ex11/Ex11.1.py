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
import random

def num_nearestDist(datafrm,feature_ID,lst_feature_types):              #numeric valuse must be normalized
    str_label_colNum = datafrm.columns[-1]
    lst_label = list((datafrm.iloc[:,-1]).drop_duplicates())          #get the list of labels
    neighborDist_UniLabel = neighborDist_DiffLabel = 0
    if len(datafrm) == 0 or lst_feature_types==[] or not(
        feature_ID in range(1,len(datafrm.columns)-1)):          #dataframe and lst_feature_types is not null,feature must within the normal range
        return 
    for itr_r in range(len(datafrm)):
        UniLabel_subset = datafrm[datafrm[str_label_colNum]==datafrm.loc[
            itr_r,str_label_colNum]]                                #get the neighbors with the same label
        DiffLabel_subset = datafrm[datafrm[str_label_colNum]!=datafrm.loc[
            itr_r,str_label_colNum]]                                #get the neighbors with the same label
        if len(UniLabel_subset) == 0:                               #no samples with the same labels
            neighborDist_UniLabel = len(datafrm)
        else:
            UniLabel_subset.drop(itr_r,axis=0,inplace = True)      #remove current sample
            if lst_feature_types[feature_ID] == 'str':
                if datafrm.iloc[itr_r,feature_ID] in UniLabel_subset.iloc[:,feature_ID].values:  #whether current sample within the same labeled subset
                    neighborDist_UniLabel = neighborDist_UniLabel - 0
                else:
                    neighborDist_UniLabel = neighborDist_UniLabel + (-1)
                if datafrm.iloc[itr_r,feature_ID] in DiffLabel_subset.iloc[:,feature_ID].values:  #whether current sample within the different labeled subset
                    neighborDist_DiffLabel = neighborDist_DiffLabel + 0
                else:
                    neighborDist_DiffLabel = neighborDist_DiffLabel + 1
            else:
                neighborDist_UniLabel = neighborDist_UniLabel - min(
                    (UniLabel_subset.iloc[:,feature_ID]-datafrm.iloc[
                        itr_r,feature_ID])**2)
                neighborDist_DiffLabel = neighborDist_DiffLabel + min(
                    (DiffLabel_subset.iloc[:,feature_ID]-datafrm.iloc[
                        itr_r,feature_ID])**2)
    return neighborDist_UniLabel+neighborDist_DiffLabel

dataset = pd.read_csv(r'D:\vs_prj\python\MLPrj\pythonPrj2026\meloneBuch\watermelon3.0.csv',encoding='gbk')
int_recNum = len(dataset)                         #Num of the total samples
int_feature = len(dataset.columns)                #Num of the features
Relief_result = dataset.iloc[0:0,:].astype(float)
print(dataset)
print(dataset.columns)
print(Relief_result)
lst_feature_type=['null']
for itr in range(1,int_feature-1):
    if isinstance(dataset.iloc[0,itr], str):
        lst_feature_type.append('str')
    else:
        lst_feature_type.append('num')    

#print(lst_feature_type)
#print(num_nearestDist(dataset,7,lst_feature_type))

for itr_c in range(1,int_feature-1):
    columnName = Relief_result.columns[itr_c]
    Relief_result.loc[0,columnName] = num_nearestDist(dataset,itr_c,lst_feature_type)
    #print(num_nearestDist(dataset,itr_c,lst_feature_type))

print(Relief_result)

