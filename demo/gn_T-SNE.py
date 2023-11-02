"""
@Create Time : 2023/7/18 
@Authors     : Allen_Chang
@Description : T-SNE可视化
@Modif. List : 230718 整体框架搭建
"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1.获取最后提取到的特征（embedding）
# 需要仿照训练或者测试代码，将数据送入模型，得到提取的特征
embedding = []
for i in range(1000):
    embedding.append(np.random.randn(1, 512))

embedding = np.array(embedding).squeeze(1)

# 2.获取标签
# 注意要与步骤一中提取到的特征向对应
label = np.random.randint(0, 100, size=[1000])
# 3.T-SNE
tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, random_state=0)
data = tsne.fit_transform(embedding)
plt.scatter(data[:, 0], data[:, 1], c=label)
plt.title('T-SNE Visualization')
plt.show()