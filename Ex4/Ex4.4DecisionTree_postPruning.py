'''基于基尼指数的分类树不剪枝算法实现'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
from pprint import PrettyPrinter
from tabulate import tabulate
import math

def showDfm(dfm, field=0, prp=0):           #输入dataform，对齐打印输出内容，保留列名：field和数据列：prp这两个调节间距的参数

    col_num = len(dfm.columns)                          #获取数据集的列数
    row_num = len(dfm)                                   #获取数据集的行数，不计入dataform的首行
    width_mtx = [[0 for _ in range(col_num)] for _ in range(row_num+1)]     #初始化宽度矩阵，行数增加一行(列名)
    width_mtx[0] = list(dfm.columns);                   #首行赋值
    width_mtx[1:]= dfm.values.tolist()                  #其余行赋值
    tmp_width = 0
    for col_idx in range(col_num):
        for row_idx in range(row_num+1):
           if isinstance(width_mtx[row_idx][col_idx], str):     #判断是否是字符型对象
                # 判断是否包含汉字
                has_chinese = any('\u4e00' <= c <= '\u9fff' for c in width_mtx[row_idx][col_idx])   #判断是否是汉字
                if has_chinese:
                    tmp_width = len(width_mtx[row_idx][col_idx]) * 2
                else:
                    tmp_width = len(width_mtx[row_idx][col_idx])
           else:
                # 数值转字符串
                val = width_mtx[row_idx][col_idx]
                if val.is_integer():                                                            #编码后的数据实际为诸如1.0，2.0的数值，需要转化后再求宽度
                    tmp_width = len(str(int(val))) 
                else:                                                                           #针对浮点数
                    tmp_width = len(str(val)) 
           width_mtx[row_idx][col_idx] = int(tmp_width)                                         #给宽度矩阵赋值为相应元素占据的宽度
        
        
    widthMax_lst = [max(row[i] for row in width_mtx) for i in range(len(width_mtx[0]))]   #找到每列的最大宽度值并放在列表中
#    for idx in range(len(widthMax_lst)): widthMax_lst[idx] = widthMax_lst[idx]+2              #微调每列的最大宽度

#    print("列宽如下")
#    print(widthMax_lst)
#显示首行：列名    
    tmp_width = 0
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
df = pd.read_csv(r'C:\Users\tanchao\pythonPrj\helloWorld\zhouEx\watermelon2.0.csv', encoding='gbk')  # 替换为你的文件路径
df_field = df.columns
df_val = df.iloc[1:,:]
print("原始数据：")
showDfm(df)

# -------------------------------
# 3. 分离特征和目标变量
# -------------------------------
X = df.drop(columns=['编号','好瓜'])  # 特征
y = df['好瓜']                 # 标签（0=坏瓜, 1=好瓜）

# -------------------------------
# 4. 对所有非数值特征进行 Label Encoding
# -------------------------------
X_encoded = X.copy()                                    #在panda库中是深层复制
label_encoders = {}

for col in X.columns:                                   #按列遍历数据集，col是列名，字符串变量
    le = LabelEncoder()
    if X[col].dtype == 'object':                        # 表示是字符串列，则进行转换编码
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le

print("\n显示编码后的特征:")
X_encoded_tmp = X_encoded.copy()
X_encoded_tmp.insert(0,"编号",df['编号'])
showDfm(X_encoded_tmp)                                 #显示前几行的样本

#idx_train, idx_test存放样本的原始索引号
X_train,X_test,y_train,y_test, idx_train, idx_test =model_selection.train_test_split(
    X_encoded, y, df['编号'], 
    test_size=0.25, 
    random_state=0, 
    stratify=y
)

# -------------------------------
# 5. 训练决策树模型
# -------------------------------
model = DecisionTreeClassifier(
    random_state=42,        #该参数是随机种⼦，⽤于控制分裂特征的随机性。
    ccp_alpha=0,            #成本复杂度参数,与树的不纯度直接相关,未剪枝时设置为0
    min_samples_leaf=1,       #预剪枝参数：叶子节点最小样本数。
    min_samples_split=2,    # 预剪枝参数：一个节点必须至少包含 2 个样本，才能尝试进行分裂
    max_depth=10,               #预剪枝参数：树分枝的最大深度 
    criterion='gini'
)
model.fit(X_train, y_train)


# 进行后剪枝操作
pruning_path = model.cost_complexity_pruning_path(X_train, y_train)  # 计算路径
#-------打印结果---------------------------    
print("\n====CCP路径=================")
print("ccp_alphas:",pruning_path['ccp_alphas'])
print("impurities:",pruning_path['impurities']) 

ccp_alphas = pruning_path.ccp_alphas[:-1]  # 去掉最后一个alpha值（对应完全展开的决策树）
clfs = []  # 存储各个子树模型
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
# 在验证集上评估每个子树模型，并选择最优的模型作为最终模型
y_preds = [clf.predict(X_test) for clf in clfs]
acc_scores = [accuracy_score(y_test, y_pred) for y_pred in y_preds]
best_clf_idx = acc_scores.index(max(acc_scores))
best_clf = clfs[best_clf_idx]
# 在测试集上评估最优模型性能
y_pred = best_clf.predict(X_test)
print("Accuracy after pruning:", accuracy_score(y_test, y_pred))

# -------------------------------
# 6. 可视化决策树（中文正常显示）
# -------------------------------
plt.figure(figsize=(20, 12))
plot_tree(model,
          feature_names=X.columns,      # 节点显示为中文列名
          class_names=['好瓜','坏瓜'],   #按照索引号对应分类结果，这里的索引0对应分类属性列的取值0，所以为'好瓜'，要与实际结果一致
          filled=True,                  #节点填充颜色
          fontsize=12,                  
          rounded=True,
          proportion=False,             #显示当前节点包含样本数量
          node_ids=True)                #显示节点序号

# 添加标题（中文）
plt.title("西瓜分类决策树", fontsize=16, pad=20)

# 显示图形
plt.show()

prediction = model.predict(X_test)
acc = accuracy_score(y_test, prediction)
print(f"\n测试集准确率: {acc:.3f} ({acc*100:.1f}%)")


print("\n测试集预测结果(含原始编号):")
print("-" * 60)

#显示预测结果，按照0代表好瓜，1代表坏瓜
for i, orgID in enumerate(idx_test):
    true_label = y_test.iloc[i]
    pred_label = prediction[i]
    correct = "✓" if true_label == pred_label else "✗"
    true_str = "好瓜" if true_label == 0 else "坏瓜"
    pred_str = "好瓜" if pred_label == 0 else "坏瓜"
    print(f"编号 {orgID+1:2d}: 实际={true_str}, 预测={pred_str} ,结果 {correct}")
    df_tmp = df.iloc[[orgID]]
    showDfm(df_tmp)
    print()
    

