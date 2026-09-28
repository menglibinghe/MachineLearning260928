'''基于基尼指数的分类树不剪枝算法实现'''
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

def showDfm(dfm, field=0, prp=0):           #输入datafram，对齐打印输出内容，保留列名field和数据列prp这两个调试间距的参数

    col_num = len(dfm.columns)                          #获取数据集的列数
    row_num = len(dfm)                                   #获取数据集的行数，不计入datafram的首行
    width_mtx = [[0 for _ in range(col_num)] for _ in range(row_num+1)]     #初始化宽度矩阵为0阵，行数增加一行(作为显示用的列名)
    width_mtx[0] = list(dfm.columns);                   #首行赋值为列名
    width_mtx[1:]= dfm.values.tolist()                  #其余行赋值
    tmp_width = 0
    for col_idx in range(col_num):
        for row_idx in range(row_num+1):
           if isinstance(width_mtx[row_idx][col_idx], str):     #判断是否是字符型对象
                # 判断是否包含汉字
                has_chinese = any('\u4e00' <= c <= '\u9fff' for c in width_mtx[row_idx][col_idx])   #判断是否是汉字
                if has_chinese:
                    tmp_width = len(str(width_mtx[row_idx][col_idx])) * 2                           #汉字会被len()截断后再识别
                else:
                    tmp_width = len(str(width_mtx[row_idx][col_idx]))
           else:
                # 数值转字符串
                val = width_mtx[row_idx][col_idx]
                if isinstance(val, int):                                                  #编码后的数据实际为诸如1.0，2.0的数值，需要转化后再求宽度
                    tmp_width = len(str(val)) 
                else:                                                                     #针对浮点数
                    tmp_width = len(str(val)) 
           width_mtx[row_idx][col_idx] = tmp_width                                        #给宽度矩阵赋值为相应元素占据的宽度
        
    #先在外层列迭代为for i in range(len(width_mtx[0]))，然后针对每一列在内层行迭代为row[i] for row in width_mtx
    widthMax_lst = [max(row[i] for row in width_mtx) for i in range(len(width_mtx[0]))]   #找到每列的最大宽度值并放在列表中
#    for idx in range(len(widthMax_lst)): widthMax_lst[idx] = widthMax_lst[idx]+2              #微调每列的最大宽度

#    print("列宽如下")
#    print(widthMax_lst)
#显示首行：列名    
    #tmp_width = 0
    for col_idx in range(col_num):
         indent = abs(widthMax_lst[col_idx]-width_mtx[0][col_idx]+field)/2
         print(" "*math.floor(indent),dfm.columns[col_idx]," "*math.ceil(indent),end="")    #通过向上，向下取整，避免列间距错位累计过快
    print()
#显示数据行    
    for row_idx in range(row_num):
        for col_idx in range(col_num):
            indent = abs(widthMax_lst[col_idx]-width_mtx[row_idx+1][col_idx]+prp)/2
            if col_idx == (col_num-1):
                print(" "*math.floor(indent),str(dfm.iloc[row_idx,col_idx])," "*math.ceil(indent),end="\n") 
            else:
                print(" "*math.floor(indent),str(dfm.iloc[row_idx,col_idx])," "*math.ceil(indent),end="") 
    

#————————————————————————————————————
#输入DataFram数据帧，获取最后一列的分类数量，返回列表：[最多样本的类别maxKey，类别总数cateNum，总信息熵totalEntropy
def count_maxCate(df):              
    rst_lst = list(df.iloc[:, -1])
    cate_dct = {}                                               #类别ID：样本数的字典，针对不同类别统计样本数量
    if rst_lst == [] or rst_lst[0] == None:                     #无分类结果列，或者结果列存在无分类的值
         print("it is empty in the results column")
         return
    cate_dct[str(rst_lst[0])] = 0
    #创建按结果列分类的字典统计样本数量
    for val in rst_lst:
        if str(val) in cate_dct.keys():
            cate_dct[str(val)] += 1
        else:
            cate_dct[str(val)] = 1                              #如果没有在字典中找到类别的键，则创建新的类别键，键值为1
    cateNum = len(cate_dct)
    #找到键值(类别样本数量)最大的类别
    maxKey = str(rst_lst[0])
    maxKeyVal = 0
    totalEntropy = 0
    for key in cate_dct:
        if cate_dct[key] > maxKeyVal:
            maxKey = key
            maxKeyVal = cate_dct[key]
        totalEntropy += -cate_dct[key]/len(rst_lst)*math.log2(cate_dct[key]/len(rst_lst))
    return [int(maxKey), cateNum, totalEntropy]



#————————————————————————————————————
#输入DataFram数据帧，导出编码后新数据集(仅针对非数值型变量进行编码，按照顺序分配0，1，2，...),omit_firstCol=1表示忽略首列(编号列)
def encode_str2num(df,omit_firstCol =1):            #输入DataFrom数据集，导出编码后新数据集(仅针对非数值型变量进行编码，按照顺序分配0，1，2，...)
    dfp = df.copy()
    if omit_firstCol != 1:                          #如果首列也要参与编码
        omit_firstCol = 0
    for col in dfp.columns[omit_firstCol:]:                     #遍历列名                  
        if any(isinstance(val,str) for val in dfp[col]):        #判断存在字符型数据的列
            col_srs = dfp[col]                                  #暂存待转换为编码的列数据(得到panda.Series类型)                                
            col_unq = dfp[col].unique()                         #求取唯一值的数组
            #查表得出每个属性值对应的编码，然后修改dfp相应的属性
            for row in range(len(col_srs)):
                for idx in range(len(col_unq)):
                    if col_srs[row] == col_unq[idx]:
                        dfp.loc[row,col] = idx                  #修改DataFrame的属性值为编码，row为行索引号，col为列名
                        break  
    return dfp

print("#################################################33")


#——————————————————————————————————
#节点生成函数, 输入上层节点和对应数据集，生成新节点，并通过成员方法predicResult加入到全局节点列表node_lst中,glbDfp是原始数据集(做查表用)
def createNode(glbDfp,upperNodeID_P=0):
    if len(defNode.node_lst) == 0:                                          #如果节点列表为空，则返回报错信息
        print('defNode.node_lst is empty')
        return
    for itr in range(len(defNode.node_lst)):
        if defNode.node_lst[itr].nodeID == upperNodeID_P:                     #定位到上层节点在节点列表的下标
            break
    if itr >= len(defNode.node_lst):                                          #节点ID定位失败
        print('节点ID搜索失败')
        return
    if len(defNode.node_lst[itr].chr_smpID_dct) == 0:                        #搜索到的上层节点中{属性值：样本编号ID}字典为空
        print('节点包含样本为空')
        return
    upperNodePos= itr
    
    for key in defNode.node_lst[upperNodePos].chr_smpID_dct:                        #遍历上层节点的（属性值：样本编号ID）字典
        IDlst = defNode.node_lst[upperNodePos].chr_smpID_dct[key]                   
        df = glbDfp[glbDfp[glbDfp.columns[0]].isin(IDlst)]                          #搜索得到符合当前属性值的样本子集，生成新的子集(新数据帧)
        df = df.drop(defNode.node_lst[upperNodePos].curSplitChrName,axis=1)         #去除上层节点分裂用的属性列
        newNode = defNode(df,upperNodeID_P)                                         #按照上层节点的属性列迭代生成新分支节点
        defNode.node_lst.append(newNode)                                            #更新全局节点列表
        if  not newNode.isLeaf and newNode.cateID == None:                          #如果新节点不是叶节点或者新节点没有类别ID，则继续生成新节点
            createNode(df,newNode.nodeID)

        
        
########################################################################################################################################    
    
#声明节点类
#分割数据集，创建层级结构，输入已编码的数据帧，返回包含所有节点的对象列表node_lst
#对象字段：节点编号nodeId、当前分裂的属性名splitChrName、上级节点编号upperNodeID、下级节点编号列表lowerNodeID_lst、是否叶节点isLeafNode
#成员方法：预测输出predicResult(好瓜为0)、统计样本数getSmpNum()、统计类别数量列表getCateNum_lst()、信息增益getCurrEntrGain()
class defNode:
    maxCateID = 0                                                #类属性：最多数量的类别ID，创建第一个节点前必须先设置
    cateNum = 0                                                  #类属性：总的分类数目，创建第一个节点前必须先设置
    node_lst = []                                                #类属性：存储节点的全局列表
    def __init__(self,dfp,upperNodeID_P=0,smpID_lstP=[]):
        
 #       self.dataSet_df = dfp.copy()
        self.nodeID = len(defNode.node_lst)                    #赋值节点编号，分配时要将上层节点里面的lowerNodeID_lst添加上当前的节点编号(建立路径关联关系)

#        self.nextSplitChrName = ''                               #下次分裂节点的属性名, 创建每一个节点的同时需要初始化
        self.upperNodeID = upperNodeID_P                         #上级节点编号
        self.lowerNodeID_lst = []                                #下级节点节点编号列表, 初始化时为空。等到下级节点创建时由下级节点进行设置，以便建立路径关联关系
        self.isLeaf = False                                       #当前节点的是否是叶节点

        
#        self.smpID_lst = smpID_lstP if smpID_lstP else list(dfp[dfp.columns[0]])  
        self.smpID_lst = list(dfp[dfp.columns[0]])                              #当前节点包含的样本编号ID,默认包括所有样本编号ID
        self.curSplitChrName,self.EntropyGain = self.getInfoGain(dfp)          #当前节点的分裂用的属性名和信息增益      
#        self.EntropyGain = self.getInfoGain(dfp)[1]              #当前节点的信息增益
        self.cateID = self.predicResult(dfp)                     #当前节点的分类结果
#存放当前节点按照属性值分开的的字典格式为，当前属性值：[样本编号ID,...]        
        self.chr_smpID_dct = self.getChr_smpID_dct(dfp,chrName=self.curSplitChrName)
        pass

      
    def predicResult(self,dfp):                                  #根据当前的节点的样本集合给出当前节点的预测结果,即类别ID
        if self.upperNodeID != self.nodeID:                      #如果当前节点不是顶级节点，则将当前节点ID加入到上层节点的子节点列表里
            defNode.node_lst[self.upperNodeID].lowerNodeID_lst.append(self.nodeID)
        if len(self.smpID_lst) == 0:                               #如果当前节点包含的样本集为空，则当前节点为叶节点，返回最大类
            self.isLeaf = True
            return defNode.maxCateID 
        
        fieldName = dfp.columns[0]
        
        selSet1 = dfp.drop_duplicates(subset = dfp.columns[-1],inplace=False)        #结果集针对类别进行去重
        if len(selSet1) == 1:                                           #如果结果集去重后，只剩下一条记录，则说明结果集的类别唯一，输入类别ID
            self.isLeaf = True
            return selSet1[selSet1.columns[-1]].iloc[0]  
        
        selSet2 = dfp.drop_duplicates(subset = dfp.columns[1:-1],inplace=True)        #结果集针对属性列进行去重
        if len(selSet1) == 1:                                           #如果结果集去重后，只剩下一条记录，则说明结果集的类别唯一，输入类别ID
            self.isLeaf = True
            return defNode.maxCateID
      
        if self.isLeaf == None:                                         #不是叶节点，预测结果为None
            return None
    #——————————————————————————————————
    def getCateNum_lst(self,dfp):                                  #为当前节点包含的样本根据不同类别统计数量，返回统计数量的列表
        fieldName = dfp.columns[0]
        cateNum_lst = [0 for _ in range(defNode.cateNum)]           #初始化统计结果列表，初始值全为0，列表长度为defNode.cateNum
        selSet_df = dfp[dfp[fieldName].isin(self.smpID_lst)]       #筛选出当前节点包含的样本子集
        df_tmp =  selSet_df.groupby(dfp.columns[-1]).count()        #针对当前节点包含的样本集按照类别分类汇总统计样本个数     
        for idx in range(len(df_tmp)):
            cateID = df_tmp.iloc[idx,0]                             #取出类别ID
            cateNum_lst[cateID] = df_tmp.iloc[idx,1]                #根据类别ID在相应的类别列表中赋值类别汇总的样本个数
        return cateNum_lst
    
    #——————————————————————————————————————————————
    #输入数据帧和列名，返回{属性值：样本编号ID列表}的字典
    def getChr_smpID_dct(self,dfp,chrName=''):                                         
        if len(dfp)*len(chrName)==0:
            return {}
        Chr_smpID_dct = {}
        col = dfp.columns.get_loc(chrName)                            #在选定的属性列里进行遍历,-1因为最后为类别列，不参与计算
        for row in range(len(dfp.iloc[:,col])):
            if not(str(dfp.iloc[row,col]) in Chr_smpID_dct):                          #如果找到属性值在字典有对应的键，则在字典的对应键值列表增加行记录编号(样本编号)
                Chr_smpID_dct[str(dfp.iloc[row,col])] =[]          
            Chr_smpID_dct[str(dfp.iloc[row,col])].append(dfp.iloc[row,0])
        return Chr_smpID_dct
    
    #————————————————————————————        
    #计算已编码好的数据帧，产生一个仅含各列属性的信息熵的数据帧，输出列表：[最大信息增益对应的列名(分裂用的属性名),最大信息增益]          
    def getInfoGain(self,df,omit_firstCol =1):
        dfp = df.copy()
        totalEntropy= count_maxCate(dfp)[2]
        dfp_entr = dfp.head(0)                                                              #存放各个数据列的信息增益的数据帧
        dfp_entr = dfp_entr.astype('float64')                                               #强制转换存储类型为浮点型
        Chr_smpID_dct = {}                                                                #针对不同属性值(键)用列表存放涵盖的样本编号的字典
        for col in range(omit_firstCol,len(dfp.columns)-1):                                 #在选定的属性列里进行遍历,-1因为最后为标签列，不参与计算。？？？原为for col in range(omit_firstCol,len(dfp.columns)-1):  
            Chr_smpID_dct = self.getChr_smpID_dct(dfp,chrName=dfp.columns[col])
            countCateNum_dct = dict.fromkeys(Chr_smpID_dct.keys())               #拷贝结构生成新字典countCateNum_dct，存放每个属性值每种分类的数量
            for key in countCateNum_dct:                                                    #初始化字典的键值列表，使列表元素数目等于类别总数，且元素值均为0
                countCateNum_dct[key] = [0 for _ in range(defNode.cateNum)]
            for key in Chr_smpID_dct: 
                for smpID in Chr_smpID_dct[key]:                           #根据当前列的不同属性值涵盖的样本统计不同分类的数量，按照类别0,1,...,存放样本数量。smpID表示样本编号
                    cateIDtmp = dfp.loc[dfp['编号'] == smpID, dfp.columns[-1]].iloc[0]    #表示根据样本编号查找样本对应的类别标签ID，再根据类别ID在列表对应位置累加计数 。
                    countCateNum_dct[key][cateIDtmp] +=1                   #得到每个属性值对应的每种类别标签下的样本统计数量
            
            
            entropy = 0
            spcSum = len(dfp)                                                               #当前属性值涵盖的样本总数
            for key in countCateNum_dct:                                                     
                p=entrTmp=0
                for itr in countCateNum_dct[key]:                                          #遍历当前属性值的每一种分类 
                    p = itr/sum(countCateNum_dct[key])
                    if p != 0:                    
                        entrTmp += -p*math.log2(p)                                          #累加每种分类对应的信息熵，得到每一种属性值的信息熵entrTmp 
                entropy += sum(countCateNum_dct[key])/spcSum*entrTmp                       #累加计算当前属性列的全部属性值的信息熵(带权重的)的总和，即属性列的信息熵
            dfp_entr.loc[0,dfp_entr.columns[col]] = totalEntropy- entropy                  #计算当前属性列的信息增益 
            Chr_smpID_dct.clear()
        maxEntGain = 0                                                                      #存储最大信息增益
        maxChrName = ''                                                                     #存储最大信息增益对应的列名
        for key in dfp_entr.columns:                                                        #遍历信息增益的各列
            if maxEntGain < dfp_entr[key].item():                                           #.item()仅针对一维数据帧有效，                                      
               maxEntGain = dfp_entr[key].item()                                           
               maxChrName = key                                                            
        return [maxChrName,maxEntGain]                                                       #返回最大信息增益对一个的列名和信息增益

#输入节点编号，返回节点在列表的位置(索引号)                                                        
def getPositonInNodeLst(node_lst, nodeID):
    for itr in range(len(defNode.node_lst)):
        if defNode.node_lst[itr].nodeID == nodeID:                     #定位到节点在节点列表的下标
            break
    return itr
    return nodeID
    
#————————————————————————————————————
#输入节点列表，返回决策树最大深度
def getTreeDepth(node_lstP,nodeID= 0):
    MaxDepth= 0                                         #暂存当前搜索树的最大深度
#    curDepth = 0                                        #记录当前搜索树的深度
    if len(node_lstP) == 0:
        print('当前节点列表为空')
        return
    pos = getPositonInNodeLst(node_lstP,nodeID)
    RootNode = node_lstP[pos]
    if len(RootNode.lowerNodeID_lst) == 0:
        print('当前根节点无子节点')
        return    
    for itr in RootNode.lowerNodeID_lst:
        childNode = node_lstP[getPositonInNodeLst(node_lstP,itr)]
        if not childNode.isLeaf:
            curDepth = 1 + getTreeDepth(node_lstP,itr)      #递归计算每条分支的深度
        else:
            curDepth = 2
        if MaxDepth < curDepth: MaxDepth = curDepth     #记录最大深度
    return MaxDepth
        
#输入节点列表，返回决策树叶节点个数
def getTreeLeafNum(node_lstP):
    leafNum = 0  
    for node in node_lstP:
        if node.isLeaf: leafNum += 1
    return leafNum      

#输入节点列表，返回决策树的二维层次分布节点编号字典，键为层次序号，键值为[当前层包含的节点编号列表]
def getHrchNodeID_dct(node_lstP):
    hrchNode_dct = {}
    TreeDept = getTreeDepth(node_lstP,nodeID= 0)
    for itr in range(TreeDept):
        hrchNode_dct[str(itr)] = []                            #根据层次数目，初始化层子字典的键值为空列表
    for node in node_lstP:
        if node.nodeID == 0:                                    #如果当前节点是根节点，在'0'键下添加[0],表示仅有根节点
            hrchNode_dct['0'].append(node.nodeID)
        else:
            TreeDept = 1
            searchNodeID =  node.upperNodeID                     #如果搜索节点不是根节点，则搜索节点号向上迭代
            while True:
                posInNodeLst= getPositonInNodeLst(node_lstP, searchNodeID)          #根据搜索节点号得到节点在节点列表的下标位置
                if node_lstP[posInNodeLst].nodeID == 0:                             #如果是两层的树，则在'1'键下添加节点号
                    hrchNode_dct[str(TreeDept)].append(node.nodeID)
                    break
                else:                                                               #否则搜索节点号继续向上迭代
                    posInNodeLst = getPositonInNodeLst(node_lstP,node_lstP[posInNodeLst].upperNodeID)
                    searchNodeID = node_lstP[posInNodeLst].nodeID
                    TreeDept += 1
    return hrchNode_dct
            
#输入节点列表和二维层次节点字典，返回按二维层次分布的字典，键为节点号：键值节点在X方向的归一化坐标列表
def getHrchNodePosXoffset_dct(node_lstP,hrchNode_dctP):
    leafNodeNum= getTreeLeafNum(node_lstP)
    hrchNodePosX_dct = deepcopy(hrchNode_dctP)
    initIndentX = 1/(leafNodeNum+1)                                                 #初始结点横向间隔为1/(所有叶节点数+1)
    for itr in range(1,len(hrchNodePosX_dct)):
        ratio= len(hrchNode_dctP[str(itr)])/len(hrchNode_dctP[str(itr-1)])
        hrchNodePosX_dct[str(itr)] = ratio*initIndentX/2 if ratio > 1 else 1               #获取调整横坐标时的间隔加倍因子，即当前层节点数/上层节点数   
    return hrchNodePosX_dct   
        
#输入节点列表和二维层次间隔列表，getHrchNodePosX_lst返回的调整间隔绘制决策树  
def plotTree(dfp,node_lstP):
    leafNodeNum= getTreeLeafNum(node_lstP)
    initIndentX = 1/(leafNodeNum+1)

    hrchNodeID_dct = getHrchNodeID_dct(node_lstP)                                       #节点按层分布的字典
    hrchNodePosXoffset_dct = getHrchNodePosXoffset_dct(node_lstP,hrchNodeID_dct)                  #节点按层设置的x方向调整间隔
    treeDept = getTreeDepth(node_lstP,0)
    leafNodeNum = getTreeLeafNum(node_lstP)

    fig, ax1 = plt.subplots(figsize=(8, 6))
 
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    IndentY = 1/(treeDept+1)                                                                 #y方向的间隔
    NodePosX_lst = [[0]]
    
    cal_changed = False
    for itr in range(1,len(hrchNodeID_dct)):                                                #按层次序号遍历
        upperNodeIDtmp_lst = []
        lstTmp =[]                                                   
        for itr1 in range(len(hrchNodeID_dct[str(itr)])):                                   #初始化每层的X坐标分布列表
            nID = hrchNodeID_dct[str(itr)][itr1]                                            #遍历每层的列表里的节点编号
            nIndex = getPositonInNodeLst(node_lstP,nID)
            upperNodeIDtmp_lst.append(node_lstP[nIndex].upperNodeID)
            if cal_changed == False:
                indentTmp = len(hrchNodeID_dct[str(itr)]) + 1
                lstTmp.append((itr1+1)/indentTmp)                                           
            if node_lstP[nIndex].isLeaf == True:                                              #
                cal_changed = True                                                            #计算方式改变标志
        NodePosX_lst.append(lstTmp)
        NodePosX_lst[1][1] += 0.02                                                          #微调1层位置
        if len(NodePosX_lst[itr]) ==0 and cal_changed == True:                                   #当前层的X坐标分布列表已有数据
            upperNodeIDtmp_lst = list(set(upperNodeIDtmp_lst))
            upperNodeIDtmp_lst.sort()
            #如果上层存在叶节点，则根据上层节点位置，排布当前节点的的X方向位置
            for itr2 in upperNodeIDtmp_lst:
                upperPosX = NodePosX_lst[itr-1][hrchNodeID_dct[str(itr-1)].index(itr2)]
                nIndex = getPositonInNodeLst(node_lstP,itr2)
                currNodeNum =len(node_lstP[nIndex].lowerNodeID_lst)
                for itr3 in range(currNodeNum):                            #按照当前层节点数的奇偶性分情况计算间隔
                    if currNodeNum % 2 ==1:
                        posXtmp = upperPosX-initIndentX*(currNodeNum-1)/2+initIndentX*(itr3)                ############
                    else:
                        posXtmp = upperPosX-initIndentX*(currNodeNum-1)/2+initIndentX*(itr3)
                    NodePosX_lst[itr].append(posXtmp)
                    NodePosX_lst[0][0] = 0.5 
            
 
            

    drawNode(0.5,treeDept/(treeDept+1),text='0#纹理',fontSizeP=8,boxType="square")      #绘制根节点

    for itr in range(1,len(hrchNodeID_dct)):
        for itr1 in range(len(hrchNodeID_dct[str(itr)])):
            nodeIndex = getPositonInNodeLst(node_lstP,hrchNodeID_dct[str(itr)][itr1])
            node = node_lstP[nodeIndex]
            boxTypeP = "ellipse" if node.isLeaf == True else "square"
            result = ''
            if node.isLeaf == True:
                if node.cateID == 0:
                    result = "好瓜"
                else:
                    result = "坏瓜"
            drawNode(NodePosX_lst[itr][itr1],(treeDept-itr)/(treeDept+1),
                     text='_'.join([str(node.nodeID),node.curSplitChrName,result]),
                     fontSizeP=8,
                     boxType=boxTypeP)
            
    plotArrow(dfp,node_lstP,hrchNodeID_dct,NodePosX_lst) 
           
    return NodePosX_lst
         
#绘制从上至下绘制箭头
def plotArrow(dfp,node_lstP,hrchNodeID_dct,NodePosX_lstP):
    treeDept = getTreeDepth(node_lstP,0)
    for itr in range(len(hrchNodeID_dct)-1):
        for itr1 in hrchNodeID_dct[str(itr)]:
            nodeIndex = getPositonInNodeLst(node_lstP,itr1)
            currNode = node_lstP[nodeIndex]
            currX = NodePosX_lstP[itr][hrchNodeID_dct[str(itr)].index(itr1)]
            currY = (treeDept-itr)/(treeDept+1)
            
            for itr2 in currNode.lowerNodeID_lst:
                itr3 = hrchNodeID_dct[str(itr+1)].index(itr2)
                childX = NodePosX_lstP[itr+1][itr3]
                childY = (treeDept-itr-1)/(treeDept+1)
                nodeIndex = getPositonInNodeLst(node_lstP,itr2)
                childNode = node_lstP[nodeIndex]
                drawArrow("text",(childX,childY),(currX,currY))
                
                nodeIndex1 = getPositonInNodeLst(node_lstP,currNode.nodeID)
                
                splitChrName = node_lstP[nodeIndex1].curSplitChrName
                smpID = node_lstP[nodeIndex].smpID_lst[0]
                plt.text((currX+childX)/2,(currY+childY)/2+0.01, str(dfp.loc[smpID,splitChrName]),ha='center', va='center', fontsize=12)
                


    
          
        
######################################################################################################
#输入节点坐标，节点文本，文本字体大小，文本框样式，调用plt子模块绘制根/叶节点
def drawNode(posX,posY,text,fontSizeP,boxType):
    plt.text(posX,posY,text,bbox=dict(boxstyle=f"{boxType},pad=0.5", facecolor="lightblue"),       
         ha='center', va='center', fontsize=fontSizeP)                                      #文本在水平和垂直方向是居中对齐
    
#输入起止节点坐标，注释文本，调用plt子模块绘制分支箭头
def drawArrow(text,Pos0_tpl,pos1_tpl):
    plt.annotate(text, xy=pos1_tpl,xytext=Pos0_tpl,                                           #xy:箭头起点（父节点）,xytext:箭头终点（子节点）
                xycoords='axes fraction',
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle="<-",
                                lw=1.5,                                                     #线条宽度
                                connectionstyle="arc3,rad=0.1",                             #弧形链接
                                color="black"),
                zorder=2)                                                                   #箭头在文本下层  
    
    

###################################################################################################################################################
# -------------------------------
# 1. 设置中文字体和负号显示
# -------------------------------

# 检查可用字体（可选，调试用）
# print(matplotlib.rcParams['font.family'])

# 设置字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 尝试这些中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号 '-'

# -------------------------------
# 2. 读取数据
# -------------------------------
df = pd.read_csv(r'C:\Users\tanchao\pythonPrj\pythonPrj2026\dataset\watermelon\watermelon2.0.csv', encoding='gbk')  # 替换为你的文件路径
df_field = df.columns
df_val = df.iloc[1:,:]
print("原始数据：")
showDfm(df)

                    
# def __init__(self,dfp,nodeID_p=0,splitChrNameP=None,upperNodeID_P=0,smpID_lstP=[]):
#分割数据集，创建层级结构，输入已编码的数据帧，返回包含所有节点的对象列表node_lst
#对象字段：节点编号nodeId、当前分裂的属性名splitChrName、上级节点编号upperNodeID、下级节点编号列表lowerNodeID_lst、是否叶节点isLeafNode
#成员方法：预测输出predicResult(好瓜为0)、统计样本数getSmpNum()、统计类别数量列表getCateNum_lst()、信息增益getCurrEntrGain()
#def __init__(self,dfp,upperNodeID_P=0,smpID_lstP=[]):

#输入DataFram数据帧，获取最后一列的分类数量，返回列表：[最多样本的类别maxKey，类别总数cateNum，总信息熵totalEntropy]

#初始化类变量            
defNode.cateNum =  count_maxCate(df)[1] 
defNode.maxCateID = count_maxCate(df)[0]
df_encoded = encode_str2num(df)

#showDfm(df_encoded)

nodeRoot = defNode(df_encoded,upperNodeID_P=0)
defNode.node_lst.append(nodeRoot)
createNode(df_encoded,0)
plotTree(df,defNode.node_lst)
plt.show()
