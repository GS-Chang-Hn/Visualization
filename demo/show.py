"""
@Create Time : 2023/6/7 
@Authors     : Allen_Chang
@Description : 请在这里添加功能描述
@Modif. List : 请在这里添加修改记录
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull

# 随机生成2000个正数
data = np.random.rand(50000)

# 使用KMeans算法将数据聚类为8个团
kmeans = KMeans(n_clusters=4).fit(data.reshape(-1, 1))
labels = kmeans.labels_

# 统计每个聚类的数量
counts = [0] * 4
for label in labels:
    counts[label] += 1

# 计算每个聚类的中心和标准差
centers = []
stds = []
for i in range(4):
    cluster_data = data[labels == i]
    center = np.mean(cluster_data)
    std = np.std(cluster_data)
    centers.append(center)
    stds.append(std)

# 计算每个聚类的凸包
hulls = []
for i in range(4):
    cluster_data = data[labels == i]
    points = np.array([(i + np.random.normal(0, 0.1), x) for x in cluster_data])
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_points = np.append(hull_points, [hull_points[0]], axis=0)
    hulls.append(hull_points)

# 绘制聚类结果图
fig, ax = plt.subplots()
for i in range(4):
    cluster_data = data[labels == i]
    x = np.random.normal(i, 0.2, size=len(cluster_data))
    y = np.random.normal(i, 0.2, size=len(cluster_data))
    ax.scatter(x, y, s=5, alpha=0.8)
    hull = plt.Polygon(hulls[i], fill=False)
    ax.add_artist(hull)

# 设置每个区域的大小
sizes = [count * 80 for count in counts]
for hull, size in zip(ax.artists, sizes):
    hull.set_linewidth(size)

plt.savefig('cluster.png')
plt.show()

