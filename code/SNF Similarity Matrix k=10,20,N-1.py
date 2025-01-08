import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

def gaussian_similarity(matrix, sigma=1.0):
    distances = pairwise_distances(matrix, metric='euclidean')
    sim_matrix = np.exp(-distances**2 / (2 * sigma**2))
    return sim_matrix

def similarity_network_fusion(connectivity_matrices, K=20, alpha=0.5, t=10, sigma=1.0):
    num_subjects = len(connectivity_matrices)
    fused_similarity_matrix = np.zeros((num_subjects, num_subjects))

    for i in range(num_subjects):
        sim_matrix_i = gaussian_similarity(connectivity_matrices[i], sigma)
        if K is not None:
            knn_graph_i = kneighbors_graph(sim_matrix_i, K, mode='connectivity', include_self=False).toarray()
            affinity_matrix_i = normalize(knn_graph_i, norm='l1', axis=1)
        else:
            knn_graph_i = kneighbors_graph(sim_matrix_i, num_subjects - 1, mode='connectivity', include_self=True).toarray()
            affinity_matrix_i = normalize(knn_graph_i, norm='l1', axis=1)
        
        for j in range(num_subjects):
            if i != j:
                sim_matrix_j = gaussian_similarity(connectivity_matrices[j], sigma)
                if K is not None:
                    knn_graph_j = kneighbors_graph(sim_matrix_j, K, mode='connectivity', include_self=False).toarray()
                    affinity_matrix_j = normalize(knn_graph_j, norm='l1', axis=1)
                else:
                    knn_graph_j = kneighbors_graph(sim_matrix_j, num_subjects - 1, mode='connectivity', include_self=True).toarray()
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

# 定义不同的 K 值
K_values = [10, 20, None]  
fused_matrices = []

for K in K_values:
    fused_matrix = similarity_network_fusion(connectivity_matrices, K=K, sigma=1.0)
    fused_matrices.append(fused_matrix)

# 可视化三个融合的相似性矩阵
plt.figure(figsize=(15, 5))

titles = ['K=10', 'K=20', 'K=N-1 (Full Connectivity)']
for i, fused_matrix in enumerate(fused_matrices):
    plt.subplot(1, 3, i + 1)
    fused_matrix_with_nan = np.copy(fused_matrix)
    np.fill_diagonal(fused_matrix_with_nan, np.nan)  # 将对角线上的值设置为 np.nan
    plt.imshow(fused_matrix_with_nan, cmap='Reds', interpolation='nearest')
    plt.title(titles[i],fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    plt.xticks(np.arange(0, num_subjects, 20), np.arange(0, num_subjects, 20),fontsize=20)
    plt.yticks(np.arange(0, num_subjects, 20), np.arange(0, num_subjects, 20),fontsize=20)
# 设置全局字体大小
plt.rcParams.update({'font.size': 20})
plt.tight_layout()
plt.show()