"""
@Create Time : 2023/6/12 
@Authors     : Allen_Chang
@Description : 请在这里添加功能描述
@Modif. List : 请在这里添加修改记录
"""
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/3/3 17:08
# @Author  : glan~
# @FileName: matrix.py
# @annotation: 6分类

import itertools
import matplotlib.pyplot as plt
import numpy as np

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        print(cm)
    else:
        print('显示具体数字：')
        print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontproperties="SimHei")
    plt.yticks(tick_marks, classes, fontproperties="SimHei")
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()

cnf_matrix = np.array([[94.3, 1.1, 0.8, 0.9, 0.4, 1.2, 0.5, 0.8],
                       [0.6, 93.1, 1.6, 0.9, 0.5, 1.2, 1.1, 0.4],
                       [0.8, 0.5, 93.9, 1.4, 0.2, 1.7, 0.6, 0.9],
                       [1.2, 0.8, 1.1, 92.8, 0.8, 0.4, 1.7, 1.2],
                       [1.3, 1.5, 0.6, 0.3, 93.7, 0.6, 0.7, 1.3],
                       [0.3, 0.9, 1.4, 0.4, 0.9, 94.5, 1.2, 0.4],
                       [0.6, 1.1, 1.1, 0.7, 0.5, 0.3, 94.7, 1.0],
                       [0.6, 0.2, 1.8, 0.3, 1.4, 1.1, 0.5, 93.7]])
attack_types = ['1', '2', '3', '4', '5', '6', '7', '8']

plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True, title='')


#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

save_flg = True

# confusion = confusion_matrix(y_test, y_pred)
confusion = np.array([[94.3, 1.1, 0.8, 0.9, 0.4, 1.2, 0.5, 0.8],
                      [0.6, 93.1, 1.6, 0.9, 0.5, 1.2, 1.1, 0.4],
                      [0.8, 0.5, 93.9, 1.4, 0.2, 1.7, 0.6, 0.9],
                      [1.2, 0.8, 1.1, 92.8, 0.8, 0.4, 1.7, 1.2],
                      [1.3, 1.5, 0.6, 0.3, 93.7, 0.6, 0.7, 1.3],
                      [0.3, 0.9, 1.4, 0.4, 0.9, 94.5, 1.2, 0.4],
                      [0.6, 1.1, 1.1, 0.7, 0.5, 0.3, 94.7, 1.0],
                      [0.6, 0.2, 1.8, 0.3, 1.4, 1.1, 0.5, 93.7]])

plt.figure(figsize=(8, 8))  #设置图片大小


# 1.热度图，后面是指定的颜色块，cmap可设置其他的不同颜色
plt.imshow(confusion, cmap=plt.cm.Blues)
plt.colorbar()   # 右边的colorbar


# 2.设置坐标轴显示列表
indices = range(len(confusion))
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
plt.xticks(indices, classes, rotation=45) # 设置横坐标方向，rotation=45为45度倾斜
plt.yticks(indices, classes)


# 3.设置全局字体
# 在本例中，坐标轴刻度和图例均用新罗马字体['TimesNewRoman']来表示
# ['SimSun']宋体；['SimHei']黑体，有很多自己都可以设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 4.设置坐标轴标题、字体
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.title('Confusion matrix')

plt.xlabel('预测值')
plt.ylabel('真实值')
plt.title('混淆矩阵', fontsize=12, fontfamily="SimHei")  #可设置标题大小、字体


# 5.显示数据
normalize = False
fmt = '.2f' if normalize else 'd'
thresh = confusion.max() / 2.

for i in range(len(confusion)):    #第几行
    for j in range(len(confusion[i])):    #第几列
        plt.text(j, i, format(confusion[i][j], fmt),
                 fontsize=16,  # 矩阵字体大小
                 horizontalalignment="center",  # 水平居中。
                 verticalalignment="center",  # 垂直居中。
                 color="white" if confusion[i, j] > thresh else "black")


#6.保存图片
# if save_flg:
#     plt.savefig("./picture/confusion_matrix.png")


# 7.显示
plt.show()