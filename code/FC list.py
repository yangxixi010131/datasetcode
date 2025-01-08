import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载.npy文件
file_path = 'all_correlation_matrices103.npy'
# 加载数据，这里假设.npy文件中存储的是一个三维数组
correlation_matrices = np.load(file_path)

# 提取第三个功能连接矩阵
third_matrix = correlation_matrices[2]  # 修改索引为2来提取第三个矩阵

# 使用Seaborn绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(third_matrix, annot=False, cmap='coolwarm')
plt.show()