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
import tensorflow as tf
from keras import layers, models

# 设置字体为 SimHei (黑体)，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] 

# 解决保存图像时负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


# # 自动下载并加载数据
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

path = r'C:\users\tanchao\pythonPrj\pythonPrj2026\meloneBuch\Ex5\mnist.npz'
with np.load(path) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

print("数据加载完成！")

# print(f"训练集形状: {x_train.shape}") # (60000, 28, 28)
# print(f"测试集形状: {x_test.shape}")  # (10000, 28, 28)
#每一张图片实际上就是一个 28行 × 28列 的数字矩阵。矩阵中的每个数字（0-255）代表该像素点的灰度值（0代表纯黑背景，255代表纯白笔迹）
# 1. 创建一个画布
plt.figure(figsize=(10, 5))         #宽 10 英寸，高 5 英寸。

# 2. 循环显示前 10 张图片
for i in range(10):
    plt.subplot(2, 5, i + 1)  # 分成 2行5列 的网格
    plt.imshow(x_train[i], cmap='gray')  # 显示图片，cmap='gray' 表示用灰度图显示
    plt.title(f"标签: {y_train[i]}")     # 标题显示真实的数字标签
    plt.axis('off')  # 关闭坐标轴

print(x_train[3])
print()
# 3. 展示图片
plt.show()



# 2. 数据预处理
# 将像素值归一化到 0-1 之间，并增加通道维度 (28, 28) -> (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0       #-1表示该处维度任意，因为Keras 的 Conv2D 层要求输入必须是4维张量 (样本数, 高, 宽, 通道数)，astype()转换整数为浮点型
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 3. 构建模型 (参考上面的参数设计)，使用Keras 的 Sequential API 搭建了一个经典的卷积神经网络
model = models.Sequential([
    # 卷积层 1，#卷积核3个，卷积窗口尺寸3*3，无padding，激活函数为修正的线性单元
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), 
    # 池化层 1，最大池化层2*2，无padding，步长为2。注意池化层深度方向不合并计算
    layers.MaxPooling2D((2, 2)),

    # 卷积层 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    # 池化层 2
    layers.MaxPooling2D((2, 2)),

    # 展平，按照通道->行序->列序的顺序进行展开和拼接
    layers.Flatten(),

    # 全连接层
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # 输出层，10个类别
])

# 4. 编译与训练
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])                     #评估指标为准确率

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)    #验证集比例为0.1

# 5. 测试
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"测试集准确率: {test_acc}")