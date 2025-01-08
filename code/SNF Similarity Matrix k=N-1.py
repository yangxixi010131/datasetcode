import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def gaussian_similarity(matrix, sigma=1.0):
    distances = pairwise_distances(matrix, metric='euclidean')
    sim_matrix = np.exp(-distances**2 / (2 * sigma**2))
    return sim_matrix

def similarity_network_fusion(connectivity_matrices, K=None, alpha=0.5, t=10, sigma=1.0):
    num_subjects = len(connectivity_matrices)
    fused_similarity_matrix = np.zeros((num_subjects, num_subjects))

    for i in range(num_subjects):
        sim_matrix_i = gaussian_similarity(connectivity_matrices[i], sigma)
        
        if K is None:
            # 创建全连接图
            knn_graph_i = kneighbors_graph(sim_matrix_i, num_subjects - 1, mode='connectivity', include_self=True).toarray()
        else:
            knn_graph_i = kneighbors_graph(sim_matrix_i, K, mode='connectivity', include_self=True).toarray()
        
        affinity_matrix_i = normalize(knn_graph_i, norm='l1', axis=1)
        
        for j in range(num_subjects):
            if i != j:
                sim_matrix_j = gaussian_similarity(connectivity_matrices[j], sigma)
                
                if K is None:
                    knn_graph_j = kneighbors_graph(sim_matrix_j, num_subjects - 1, mode='connectivity', include_self=True).toarray()
                else:
                    knn_graph_j = kneighbors_graph(sim_matrix_j, K, mode='connectivity', include_self=True).toarray()
                
                affinity_matrix_j = normalize(knn_graph_j, norm='l1', axis=1)
                
                similarity = np.mean(np.minimum(affinity_matrix_i, affinity_matrix_j))
                fused_similarity_matrix[i, j] = similarity
                fused_similarity_matrix[j, i] = similarity
            else:
                fused_similarity_matrix[i, j] = 1

    return fused_similarity_matrix

# 加载数据
all_correlation_matrices_np = np.load('all_correlation_matrices103.npy')
if np.any(np.isnan(all_correlation_matrices_np)):
    mean_value = np.nanmean(all_correlation_matrices_np)
    all_correlation_matrices_np[np.isnan(all_correlation_matrices_np)] = mean_value

num_subjects = all_correlation_matrices_np.shape[0]
connectivity_matrices = [all_correlation_matrices_np[i] for i in range(num_subjects)]

# 应用 SNF 融合相似性网络
fused_similarity_matrix = similarity_network_fusion(connectivity_matrices, K=None, sigma=1.0)

# 保存融合的相似性矩阵
np.save('fused_similarity_matrix_kN-1.npy', fused_similarity_matrix)

# 可视化融合的相似性矩阵
plt.figure(figsize=(8, 6))
fused_matrix_with_nan = np.copy(fused_similarity_matrix)
np.fill_diagonal(fused_matrix_with_nan, np.nan)  # 将对角线上的值设置为 np.nan
plt.imshow(fused_matrix_with_nan, cmap='Reds', interpolation='nearest')
plt.title('K=N-1')
plt.colorbar()
plt.xlabel('Subject')
plt.ylabel('Subject')
plt.xticks(np.arange(0, num_subjects, 20), np.arange(0, num_subjects, 20))
plt.yticks(np.arange(0, num_subjects, 20), np.arange(0, num_subjects, 20))
plt.tight_layout()
plt.show()