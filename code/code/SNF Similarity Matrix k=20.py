import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


def gaussian_similarity(matrix, sigma=1.0):
    """
    Compute Gaussian similarity matrix
    """
    distances = pairwise_distances(matrix, metric='euclidean')
    sim_matrix = np.exp(-distances ** 2 / (2 * sigma ** 2))
    return sim_matrix


def similarity_network_fusion(connectivity_matrices, K=20, alpha=0.5, t=10, sigma=1.0):
    """
    Implementation of Similarity Network Fusion (SNF) algorithm
    """
    num_subjects = len(connectivity_matrices)
    fused_similarity_matrix = np.zeros((num_subjects, num_subjects))

    # Step 1: Build K-nearest neighbor graph
    for i in range(num_subjects):
        for j in range(num_subjects):
            if i != j:
                # Compute Gaussian similarity matrix
                sim_matrix_i = gaussian_similarity(connectivity_matrices[i], sigma)
                sim_matrix_j = gaussian_similarity(connectivity_matrices[j], sigma)

                # Get K-nearest neighbor graph
                knn_graph_i = kneighbors_graph(sim_matrix_i, K, mode='connectivity', include_self=False).toarray()
                knn_graph_j = kneighbors_graph(sim_matrix_j, K, mode='connectivity', include_self=False).toarray()

                # Normalize affinity matrix
                affinity_matrix_i = normalize(knn_graph_i, norm='l1', axis=1)
                affinity_matrix_j = normalize(knn_graph_j, norm='l1', axis=1)

                # Compute similarity between two subjects
                similarity = np.mean(np.minimum(affinity_matrix_i, affinity_matrix_j))
                fused_similarity_matrix[i, j] = similarity
                fused_similarity_matrix[j, i] = similarity

    return fused_similarity_matrix


# Load your functional connectivity matrices, assumed to be a 3D numpy array
all_correlation_matrices_np = np.load('all_correlation_matrices103.npy')

# Check for NaN values
if np.any(np.isnan(all_correlation_matrices_np)):
    # Replace NaN values with mean
    mean_value = np.nanmean(all_correlation_matrices_np)
    all_correlation_matrices_np[np.isnan(all_correlation_matrices_np)] = mean_value

# Reshape all_correlation_matrices_np into a list of 2D numpy arrays (connectivity matrices)
num_subjects = all_correlation_matrices_np.shape[0]  # Number of subjects
connectivity_matrices = [all_correlation_matrices_np[i] for i in range(num_subjects)]

# Apply SNF to fuse similarity networks
fused_similarity_matrix = similarity_network_fusion(connectivity_matrices, sigma=1.0)

# Save the fused similarity matrix
np.save('fused_similarity_matrix_k20.npy', fused_similarity_matrix)

# Print the fused similarity matrix
# print("Fused similarity matrix:")
# print(fused_similarity_matrix)

# Visualize the fused similarity matrix
plt.figure(figsize=(8, 6))
# Adjust color mapping range with vmin and vmax
plt.imshow(fused_similarity_matrix, cmap='Reds', interpolation='nearest')
plt.title('K=20')
plt.colorbar()
plt.xlabel('Subject')
plt.ylabel('Subject')
plt.tight_layout()
plt.show()