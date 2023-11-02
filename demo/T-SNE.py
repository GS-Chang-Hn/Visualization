"""
@Create Time : 2023/6/25 
@Authors     : Allen_Chang
@Description : 请在这里添加功能描述
@Modif. List : 请在这里添加修改记录
"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 生成两个随机的tensor
tensor1 = np.random.randn(1, 512)
tensor2 = np.random.randn(1, 512)

# 合并两个tensor
combined_tensor = np.concatenate((tensor1, tensor2), axis=0).tolist()

# 使用TSNE对合并的tensor进行降维可视化
tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0)
tsne_results = tsne.fit_transform(combined_tensor)

# 分别绘制两个tensor的点
plt.scatter(tsne_results[0, 0], tsne_results[0, 1], label='Tensor 1')
plt.scatter(tsne_results[1, 0], tsne_results[1, 1], label='Tensor 2')

# 添加图例、标题等
plt.legend()
plt.title('T-SNE Visualization of Two Tensors')
plt.show()