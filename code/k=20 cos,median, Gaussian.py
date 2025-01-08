import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# 定义不同的相似性矩阵计算函数
def cosine_similarity_matrix(matrix):
    return cosine_similarity(matrix)

def median_similarity_matrix(matrix):
    distances = pairwise_distances(matrix, metric='euclidean')
    median_distance = np.median(distances)
    return np.exp(-distances / median_distance)

def gaussian_similarity_matrix(matrix, sigma=1.0):
    distances = pairwise_distances(matrix, metric='euclidean')
    return np.exp(-distances ** 2 / (2 * sigma ** 2))

# 定义相似性网络融合函数
def similarity_network_fusion(connectivity_matrices, similarity_function, K=20, alpha=0.5, t=10, **kwargs):
    num_subjects = len(connectivity_matrices)
    fused_similarity_matrix = np.zeros((num_subjects, num_subjects))

    for i in range(num_subjects):
        for j in range(num_subjects):
            if i != j:
                sim_matrix_i = similarity_function(connectivity_matrices[i], **kwargs)
                sim_matrix_j = similarity_function(connectivity_matrices[j], **kwargs)
                
                knn_graph_i = kneighbors_graph(sim_matrix_i, K, mode='connectivity', include_self=False).toarray()
                knn_graph_j = kneighbors_graph(sim_matrix_j, K, mode='connectivity', include_self=False).toarray()
                
                affinity_matrix_i = normalize(knn_graph_i, norm='l1', axis=1)
                affinity_matrix_j = normalize(knn_graph_j, norm='l1', axis=1)
                
                similarity = np.mean(np.minimum(affinity_matrix_i, affinity_matrix_j))
                fused_similarity_matrix[i, j] = similarity
                fused_similarity_matrix[j, i] = similarity

    return fused_similarity_matrix

# 加载数据
all_correlation_matrices_np = np.load('all_correlation_matrices103.npy')
num_subjects = all_correlation_matrices_np.shape[0]
connectivity_matrices = [all_correlation_matrices_np[i] for i in range(num_subjects)]

# 检查 NaN 值
if np.any(np.isnan(all_correlation_matrices_np)):
    mean_value = np.nanmean(all_correlation_matrices_np)
    all_correlation_matrices_np[np.isnan(all_correlation_matrices_np)] = mean_value

num_subjects = all_correlation_matrices_np.shape[0]
connectivity_matrices = [all_correlation_matrices_np[i] for i in range(num_subjects)]

# 计算相似性矩阵
fused_cosine = similarity_network_fusion(connectivity_matrices, cosine_similarity_matrix, K=20)
fused_median = similarity_network_fusion(connectivity_matrices, median_similarity_matrix, K=20)
fused_gaussian = similarity_network_fusion(connectivity_matrices, gaussian_similarity_matrix, K=20, sigma=1.0)

# 绘制合并的直方图和多项式拟合曲线
fig, ax = plt.subplots(figsize=(10, 6))

def plot_edge_weight_distribution_with_curve(matrix, label, color, ax, degree=3):
    edge_weights = matrix[np.triu_indices_from(matrix, k=1)]
    hist, bin_edges = np.histogram(edge_weights, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 绘制直方图
    ax.hist(edge_weights, bins=100, alpha=0.7, label=label, color=color, edgecolor='black', density=True)

    # 拟合多项式曲线
    if len(bin_centers) > 2:
        coeffs = np.polyfit(bin_centers, hist, degree)
        poly = np.poly1d(coeffs)
        x = np.linspace(min(bin_centers), max(bin_centers), 100)
        ax.plot(x, poly(x), color=color, linestyle='-', linewidth=2, label=label)

    ax.set_xlabel('Edge Weights',fontsize=20)
    ax.set_ylabel('Edge Numbers',fontsize=20)
    ax.set_title('Edge Weight Distribution with Curve',fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True)

plot_edge_weight_distribution_with_curve(fused_cosine, 'Cosine Similarity', 'r', ax)
plot_edge_weight_distribution_with_curve(fused_median, 'Median Similarity', 'g', ax)
plot_edge_weight_distribution_with_curve(fused_gaussian, 'Gaussian Similarity', 'b', ax)
print("Shape of fused_cosine matrix:", fused_cosine.shape)
print("Shape of fused_median matrix:", fused_median.shape)
print("Shape of fused_gaussian matrix:", fused_gaussian.shape)
# 设置全局字体大小
plt.rcParams.update({'font.size': 20})

# 调整坐标轴刻度标签的字体大小
ax.tick_params(axis='both', which='major', labelsize=20)
plt.tight_layout()
plt.show()