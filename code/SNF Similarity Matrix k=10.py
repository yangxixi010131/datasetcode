import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def gaussian_similarity(matrix, sigma=1.0):
    """
    计算 Gaussian 相似性矩阵
    """
    distances = pairwise_distances(matrix, metric='euclidean')
    sim_matrix = np.exp(-distances**2 / (2 * sigma**2))
    return sim_matrix

def similarity_network_fusion(connectivity_matrices, K=10, alpha=0.5, t=10, sigma=1.0):
    """
    相似性网络融合（SNF）算法实现
    """
    num_subjects = len(connectivity_matrices)
    fused_similarity_matrix = np.zeros((num_subjects, num_subjects))

    # 第一步：构建K近邻图
    for i in range(num_subjects):
        for j in range(num_subjects):
            if i != j:
                # 计算 Gaussian 相似性矩阵
                sim_matrix_i = gaussian_similarity(connectivity_matrices[i], sigma)
                sim_matrix_j = gaussian_similarity(connectivity_matrices[j], sigma)
                
                # 获取K近邻图
                knn_graph_i = kneighbors_graph(sim_matrix_i, K, mode='connectivity', include_self=False).toarray()
                knn_graph_j = kneighbors_graph(sim_matrix_j, K, mode='connectivity', include_self=False).toarray()
                
                # 归一化亲和矩阵
                affinity_matrix_i = normalize(knn_graph_i, norm='l1', axis=1)
                affinity_matrix_j = normalize(knn_graph_j, norm='l1', axis=1)
                
                # 计算两个受试者之间的相似度
                similarity = np.mean(np.minimum(affinity_matrix_i, affinity_matrix_j))
                fused_similarity_matrix[i, j] = similarity
                fused_similarity_matrix[j, i] = similarity

    return fused_similarity_matrix

# 加载你的功能连接矩阵，假设它是一个三维numpy数组
all_correlation_matrices_np = np.load('all_correlation_matrices103.npy')

# 检查 NaN 值
if np.any(np.isnan(all_correlation_matrices_np)):
    # 用均值替换 NaN 值
    mean_value = np.nanmean(all_correlation_matrices_np)
    all_correlation_matrices_np[np.isnan(all_correlation_matrices_np)] = mean_value

# 将 all_correlation_matrices_np 重塑为二维numpy数组列表（连接矩阵）
num_subjects = all_correlation_matrices_np.shape[0]  # 受试者数量
connectivity_matrices = [all_correlation_matrices_np[i] for i in range(num_subjects)]

# 应用 SNF 融合相似性网络
fused_similarity_matrix = similarity_network_fusion(connectivity_matrices, sigma=1.0)

# 保存融合的相似性矩阵
np.save('fused_similarity_matrix_k10.npy', fused_similarity_matrix)

# 打印融合的相似性矩阵
# print("融合的相似性矩阵：")
# print(fused_similarity_matrix)

# 可视化融合的相似性矩阵
plt.figure(figsize=(8, 6))
# 使用vmin和vmax调整颜色映射的范围
plt.imshow(fused_similarity_matrix, cmap='Reds', interpolation='nearest')
plt.title('K=10')
plt.colorbar()
plt.xlabel('Subject')
plt.ylabel('Subject')
plt.tight_layout()
plt.show()