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


'''
Randomly select the initial mean_vector list lst_InitMeanVec, 
covariance matrix corMat = [0.1 0; 0 0.1], 
the number of Gaussian cluster Gc_num = 3, 
'''


covarMat = np.tile(np.array([[0.1,0],[0,0.1]]),(3,1,1))    #covariance matrix(3D), index order should be given from outer to inner, the outmost corresponds to Gaussian component
Gc_num = 3                                      #Num of the mix Gaussian
dataset = pd.read_csv(r'D:\vs_prj\python\MLPrj\pythonPrj2026\meloneBuch\watermelon4.0.csv',encoding='gbk')
int_recNum = len(dataset)                       #Num of the total samples
int_featNum = len(dataset.iloc[0,:])-1          #Num of data features
print(dataset)
print(dataset.columns)
epoch = 0

lst_mixCoe=[]                                   #Initialize the Gaussian mixture coefficients
array_meanVector = np.array([])
array_postPrb = np.zeros((int_recNum,Gc_num))   #Initialize the posterior probabilities of Gaussian mixtures,each row corresponds to a sample
        

for epoch in range(Gc_num):
    lst_mixCoe.append(1/Gc_num)
meanVecTmp = np.array(dataset.sample(n=Gc_num))
array_meanVector = np.transpose(meanVecTmp[:,1:])        #get the array of mean vectors, each column is a sampled x, ['密度','含糖量']ᵀ


while epoch < 5:
    for itr_r in range(int_recNum):                                     #遍历数据(行记录)
        alpha_p_x = np.array([[0.0]])
        array_xVec = np.reshape(dataset.iloc[itr_r,1:].tolist(),(2,1))    #get the current data vector
        for itr_G1 in range(Gc_num):
            alpha_p_x = alpha_p_x + lst_mixCoe[itr_G1]*1/2/np.pi/math.sqrt(np.linalg.det(covarMat[itr_G1,:,:]))*np.exp(-1/2*
                np.transpose(array_xVec-array_meanVector[:,itr_G1:itr_G1+1])@np.linalg.inv(covarMat[itr_G1,:,:])
                @(array_xVec-array_meanVector[:,itr_G1:itr_G1+1]))
            
        for itr_G in range(Gc_num):                                     #遍历高斯分量个数
            if np.linalg.det(covarMat[itr_G,:,:]) == 0:
                print('covariance matrix has a zero determinant')
                break
            
            alpha_p_x_j = lst_mixCoe[itr_G]/2/np.pi/math.sqrt(np.linalg.det(covarMat[itr_G,:,:]))*np.exp(-1/2*
                    np.transpose((array_xVec-array_meanVector[:,itr_G:itr_G+1]))@np.linalg.inv(covarMat[itr_G,:,:])
                    @(array_xVec-array_meanVector[:,itr_G:itr_G+1]))
            array_postPrb[itr_r,itr_G] = alpha_p_x_j.item()/alpha_p_x.item()    #compute Posterior probabilities
    
    array_x = np.reshape(dataset.iloc[:,1:],(int_recNum,int_featNum))       #get the total samples array
    for itr_G in range(Gc_num): 
        Vec_tmp1 = np.sum(array_x*array_postPrb[:,itr_G:itr_G+1],axis=0)    #summation of all weighted posterior probabilities
        Vec_tmp2 = np.sum(array_postPrb[:,itr_G:itr_G+1],axis=0)            #summation of all posterior probabilities                                 #summation into a row vector as Denominator
        array_meanVector[:,itr_G:itr_G+1] = np.reshape(Vec_tmp1/Vec_tmp2,(int_featNum,1))         #update mean vectors
        
        array_postPrb_element = np.zeros((int_featNum,int_featNum))           #initialize the accumulator for each covariance matrix
        for itr_r in range(int_recNum):
            array_postPrb_element = array_postPrb_element + array_postPrb[itr_r,itr_G]*(
                np.transpose(array_x[itr_r:itr_r+1,:])-array_meanVector[:,itr_G:itr_G+1]
                )@(array_x[itr_r:itr_r+1,:]-np.transpose(array_meanVector[:,itr_G:itr_G+1]))
        covarMat[itr_G:itr_G+1,:,:] = array_postPrb_element/Vec_tmp2                                #update covariance matrix
        
        lst_mixCoe[itr_G] = Vec_tmp2/int_recNum                                                     #update mixture coefficients
    epoch +=1

Id_cluster = -1
dataset_clustered = dataset.copy(deep=True) 
dataset_clustered['clusterId']=-1
for itr_r in range(int_recNum):
    lst_recTmp = array_postPrb[itr_r,:]
    Id_cluster = [itr_G for itr_G in range(len(lst_recTmp)) if max(lst_recTmp) == lst_recTmp[itr_G]]
    dataset_clustered.loc[itr_r,'clusterId'] = Id_cluster

#print(dataset_clustered)

#数据预处理
# X = dataset_clustered[['密度','含糖率']]     #选择密度和含糖率两列
# Y = dataset_clustered['clusterId']                 #选择簇标列
df_cluster0 = dataset_clustered[dataset_clustered['clusterId'] == 0]  #选择簇标为0的行记录
df_cluster1 = dataset_clustered[dataset_clustered['clusterId'] == 1]  #选择簇标为1的行记录
df_cluster2 = dataset_clustered[dataset_clustered['clusterId'] == 2]  #选择簇标为2的行记录

#画图
f1 = plt.figure()              #创建空白图表
plt.title('watermelon_4.0')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.xlim(0,1)                   #设置x轴范围
plt.ylim(0,1)                   #设置y轴范围
plt.scatter(df_cluster0['密度'],df_cluster0['含糖率'],marker='x',color='r',s=50,label='cluster0')   #s表示描点的大小，设定横轴为密度，纵轴为含糖量
plt.scatter(df_cluster1['密度'],df_cluster1['含糖率'],marker='^',color='g',s=50,label='cluster1')
plt.scatter(df_cluster2['密度'],df_cluster2['含糖率'],marker='*',color='y',s=50,label='cluster2')

plt.scatter(array_meanVector[0:1,:],array_meanVector[1:2,:],marker='o',color='k',s=50,label='mean vector')
print(array_meanVector)
plt.legend(loc='upper right')

plt.show(block=False)

###########################################################

while epoch < 25:
    for itr_r in range(int_recNum):                                     #遍历数据(行记录)
        alpha_p_x = np.array([[0.0]])
        array_xVec = np.reshape(dataset.iloc[itr_r,1:].tolist(),(2,1))    #get the current data vector
        for itr_G1 in range(Gc_num):
            alpha_p_x = alpha_p_x + lst_mixCoe[itr_G1]*1/2/np.pi/math.sqrt(np.linalg.det(covarMat[itr_G1,:,:]))*np.exp(-1/2*
                np.transpose(array_xVec-array_meanVector[:,itr_G1:itr_G1+1])@np.linalg.inv(covarMat[itr_G1,:,:])
                @(array_xVec-array_meanVector[:,itr_G1:itr_G1+1]))
            
        for itr_G in range(Gc_num):                                     #遍历高斯分量个数
            if np.linalg.det(covarMat[itr_G,:,:]) == 0:
                print('covariance matrix has a zero determinant')
                break
            
            alpha_p_x_j = lst_mixCoe[itr_G]/2/np.pi/math.sqrt(np.linalg.det(covarMat[itr_G,:,:]))*np.exp(-1/2*
                    np.transpose((array_xVec-array_meanVector[:,itr_G:itr_G+1]))@np.linalg.inv(covarMat[itr_G,:,:])
                    @(array_xVec-array_meanVector[:,itr_G:itr_G+1]))
            array_postPrb[itr_r,itr_G] = alpha_p_x_j.item()/alpha_p_x.item()    #compute Posterior probabilities
    
    array_x = np.reshape(dataset.iloc[:,1:],(int_recNum,int_featNum))       #get the total samples array
    for itr_G in range(Gc_num): 
        Vec_tmp1 = np.sum(array_x*array_postPrb[:,itr_G:itr_G+1],axis=0)    #summation of all weighted posterior probabilities
        Vec_tmp2 = np.sum(array_postPrb[:,itr_G:itr_G+1],axis=0)            #summation of all posterior probabilities                                 #summation into a row vector as Denominator
        array_meanVector[:,itr_G:itr_G+1] = np.reshape(Vec_tmp1/Vec_tmp2,(int_featNum,1))         #update mean vectors
        
        array_postPrb_element = np.zeros((int_featNum,int_featNum))           #initialize the accumulator for each covariance matrix
        for itr_r in range(int_recNum):
            array_postPrb_element = array_postPrb_element + array_postPrb[itr_r,itr_G]*(
                np.transpose(array_x[itr_r:itr_r+1,:])-array_meanVector[:,itr_G:itr_G+1]
                )@(array_x[itr_r:itr_r+1,:]-np.transpose(array_meanVector[:,itr_G:itr_G+1]))
        covarMat[itr_G:itr_G+1,:,:] = array_postPrb_element/Vec_tmp2                                #update covariance matrix
        
        lst_mixCoe[itr_G] = Vec_tmp2/int_recNum                                                     #update mixture coefficients
    epoch +=1

Id_cluster = -1
dataset_clustered = dataset.copy(deep=True) 
dataset_clustered['clusterId']=-1
for itr_r in range(int_recNum):
    lst_recTmp = array_postPrb[itr_r,:]
    Id_cluster = [itr_G for itr_G in range(len(lst_recTmp)) if max(lst_recTmp) == lst_recTmp[itr_G]]
    dataset_clustered.loc[itr_r,'clusterId'] = Id_cluster

#print(dataset_clustered)

#数据预处理
# X = dataset_clustered[['密度','含糖率']]     #选择密度和含糖率两列
# Y = dataset_clustered['clusterId']                 #选择簇标列
df_cluster0 = dataset_clustered[dataset_clustered['clusterId'] == 0]  #选择簇标为0的行记录
df_cluster1 = dataset_clustered[dataset_clustered['clusterId'] == 1]  #选择簇标为1的行记录
df_cluster2 = dataset_clustered[dataset_clustered['clusterId'] == 2]  #选择簇标为2的行记录

#画图
f2 = plt.figure()              #创建空白图表
plt.title('watermelon_4.0')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.xlim(0,1)                   #设置x轴范围
plt.ylim(0,1)                   #设置y轴范围
plt.scatter(df_cluster0['密度'],df_cluster0['含糖率'],marker='x',color='r',s=50,label='cluster0')   #s表示描点的大小，设定横轴为密度，纵轴为含糖量
plt.scatter(df_cluster1['密度'],df_cluster1['含糖率'],marker='^',color='g',s=50,label='cluster1')
plt.scatter(df_cluster2['密度'],df_cluster2['含糖率'],marker='*',color='y',s=50,label='cluster2')

plt.scatter(array_meanVector[0:1,:],array_meanVector[1:2,:],marker='o',color='k',s=50,label='mean vector')
print(array_meanVector)
plt.legend(loc='upper right')

plt.show(block=False)

################################################


while epoch < 45:
    for itr_r in range(int_recNum):                                     #遍历数据(行记录)
        alpha_p_x = np.array([[0.0]])
        array_xVec = np.reshape(dataset.iloc[itr_r,1:].tolist(),(2,1))    #get the current data vector
        for itr_G1 in range(Gc_num):
            alpha_p_x = alpha_p_x + lst_mixCoe[itr_G1]*1/2/np.pi/math.sqrt(np.linalg.det(covarMat[itr_G1,:,:]))*np.exp(-1/2*
                np.transpose(array_xVec-array_meanVector[:,itr_G1:itr_G1+1])@np.linalg.inv(covarMat[itr_G1,:,:])
                @(array_xVec-array_meanVector[:,itr_G1:itr_G1+1]))
            
        for itr_G in range(Gc_num):                                     #遍历高斯分量个数
            if np.linalg.det(covarMat[itr_G,:,:]) == 0:
                print('covariance matrix has a zero determinant')
                break
            
            alpha_p_x_j = lst_mixCoe[itr_G]/2/np.pi/math.sqrt(np.linalg.det(covarMat[itr_G,:,:]))*np.exp(-1/2*
                    np.transpose((array_xVec-array_meanVector[:,itr_G:itr_G+1]))@np.linalg.inv(covarMat[itr_G,:,:])
                    @(array_xVec-array_meanVector[:,itr_G:itr_G+1]))
            array_postPrb[itr_r,itr_G] = alpha_p_x_j.item()/alpha_p_x.item()    #compute Posterior probabilities
    
    array_x = np.reshape(dataset.iloc[:,1:],(int_recNum,int_featNum))       #get the total samples array
    for itr_G in range(Gc_num): 
        Vec_tmp1 = np.sum(array_x*array_postPrb[:,itr_G:itr_G+1],axis=0)    #summation of all weighted posterior probabilities
        Vec_tmp2 = np.sum(array_postPrb[:,itr_G:itr_G+1],axis=0)            #summation of all posterior probabilities                                 #summation into a row vector as Denominator
        array_meanVector[:,itr_G:itr_G+1] = np.reshape(Vec_tmp1/Vec_tmp2,(int_featNum,1))         #update mean vectors
        
        array_postPrb_element = np.zeros((int_featNum,int_featNum))           #initialize the accumulator for each covariance matrix
        for itr_r in range(int_recNum):
            array_postPrb_element = array_postPrb_element + array_postPrb[itr_r,itr_G]*(
                np.transpose(array_x[itr_r:itr_r+1,:])-array_meanVector[:,itr_G:itr_G+1]
                )@(array_x[itr_r:itr_r+1,:]-np.transpose(array_meanVector[:,itr_G:itr_G+1]))
        covarMat[itr_G:itr_G+1,:,:] = array_postPrb_element/Vec_tmp2                                #update covariance matrix
        
        lst_mixCoe[itr_G] = Vec_tmp2/int_recNum                                                     #update mixture coefficients
    epoch +=1

Id_cluster = -1
dataset_clustered = dataset.copy(deep=True) 
dataset_clustered['clusterId']=-1
for itr_r in range(int_recNum):
    lst_recTmp = array_postPrb[itr_r,:]
    Id_cluster = [itr_G for itr_G in range(len(lst_recTmp)) if max(lst_recTmp) == lst_recTmp[itr_G]]
    dataset_clustered.loc[itr_r,'clusterId'] = Id_cluster

#print(dataset_clustered)

#数据预处理
# X = dataset_clustered[['密度','含糖率']]     #选择密度和含糖率两列
# Y = dataset_clustered['clusterId']                 #选择簇标列
df_cluster0 = dataset_clustered[dataset_clustered['clusterId'] == 0]  #选择簇标为0的行记录
df_cluster1 = dataset_clustered[dataset_clustered['clusterId'] == 1]  #选择簇标为1的行记录
df_cluster2 = dataset_clustered[dataset_clustered['clusterId'] == 2]  #选择簇标为2的行记录

#画图
f3 = plt.figure()              #创建空白图表
plt.title('watermelon_4.0')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.xlim(0,1)                   #设置x轴范围
plt.ylim(0,1)                   #设置y轴范围
plt.scatter(df_cluster0['密度'],df_cluster0['含糖率'],marker='x',color='r',s=50,label='cluster0')   #s表示描点的大小，设定横轴为密度，纵轴为含糖量
plt.scatter(df_cluster1['密度'],df_cluster1['含糖率'],marker='^',color='g',s=50,label='cluster1')
plt.scatter(df_cluster2['密度'],df_cluster2['含糖率'],marker='*',color='y',s=50,label='cluster2')

plt.scatter(array_meanVector[0:1,:],array_meanVector[1:2,:],marker='o',color='k',s=50,label='mean vector')
print(array_meanVector)
plt.legend(loc='upper right')

plt.show(block=False)
plt.show()