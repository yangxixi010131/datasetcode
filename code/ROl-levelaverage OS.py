import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 2)
        self.norm1 = torch.nn.BatchNorm1d(128)
        self.norm2 = torch.nn.BatchNorm1d(64)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

def block_region(matrix, region, brain_regions):
    indices = brain_regions[region]
    blocked_matrix = matrix.copy()
    if matrix.ndim == 3:  # 检查是否为三维数组
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i in indices or j in indices:
                    blocked_matrix[i, j] = 0
    elif matrix.ndim == 2:  # 二维数组
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i in indices or j in indices:
                    blocked_matrix[i, j] = 0
    return blocked_matrix

# 数据加载函数
def load_data():
    matrices = np.load('all_correlation_matrices103.npy')
    labels_df = pd.read_csv('PPMI103.txt', sep='\t', header=None)
    labels = labels_df[3].apply(lambda x: 0 if x == 'PD' else 1).values

    if np.isnan(matrices).any():
        matrices = np.nan_to_num(matrices)

    if pd.isna(labels).any():
        labels = np.nan_to_num(labels)

    num_subjects = matrices.shape[0]
    num_features = matrices.shape[1] * matrices.shape[2]

    scaler = StandardScaler()
    matrices = matrices.reshape(num_subjects, -1)
    matrices = scaler.fit_transform(matrices)

    similarity_matrix = np.load('fused_similarity_matrix_k20.npy')
    edge_index = np.array(np.nonzero(similarity_matrix > 0.5))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    x = torch.tensor(matrices, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    return x, edge_index, y, num_features

# 创建数据对象和划分训练集/测试集
def create_data(x, edge_index, y):
    num_nodes = x.size(0)
    indices = list(range(num_nodes))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=y.numpy())

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    test_mask[test_indices] = True

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask

    return data

# 训练函数
def train(model, data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

# 测试函数
def test(model, data, device):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out[data.test_mask].argmax(dim=1)
        accuracy = accuracy_score(data.y[data.test_mask].cpu().numpy(), pred.cpu().numpy())
        return accuracy

# 主函数
def main():
    x, edge_index, y, num_features = load_data()
    data = create_data(x, edge_index, y)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_node_features=num_features).to(device)
    data = data.to(device)

    train(model, data, device)
    full_accuracy = test(model, data, device)

    brain_regions = {
        "Frontal": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33],
        "Occipital": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
        "Parietal": [34, 35, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
        "Subcortical": [28, 29, 70, 71, 72, 73, 74, 75, 76, 77],
        "Temporal": [36, 37, 38, 39, 40, 41, 54, 55, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
        "Cerebellum": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
    }

    accuracies = {region: [] for region in brain_regions.keys()}

    for region in brain_regions.keys():
        for _ in range(10):  # Repeat multiple times to enhance the reliability of the results
            X_blocked = np.array([block_region(matrix.reshape(116, 116), region, brain_regions) for matrix in data.x.cpu().numpy()])
            X_blocked = X_blocked.reshape(-1, 116*116)  # 将三维矩阵转换回二维张量

            scaler = StandardScaler()
            X_blocked = scaler.fit_transform(X_blocked)  # 标准化数据
            x_blocked = torch.tensor(X_blocked, dtype=torch.float).to(device)

            data.x = x_blocked
            train(model, data, device)
            accuracy = test(model, data, device)
            accuracies[region].append(accuracy)

    mean_accuracies = {region: np.mean(accs) for region, accs in accuracies.items()}
    delta_accuracies = {region: abs(full_accuracy - mean_acc) for region, mean_acc in mean_accuracies.items()}  # 取绝对值

    delta_accuracy_df = pd.DataFrame(list(delta_accuracies.items()), columns=['Region', 'Δ Accuracy'])

    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    plt.bar(delta_accuracy_df['Region'], delta_accuracy_df['Δ Accuracy'], width=bar_width, color=sns.color_palette('viridis', len(delta_accuracy_df)))
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)  # 调整y轴刻度字体大小
    plt.title('ROI-level average occlusion sensitivity',fontsize=20)
    plt.xlabel('Brain lobes',fontsize=20)
    plt.ylabel('Δ Accuracy per ROI',fontsize=20)
    plt.tight_layout()
    plt.show()
   
if __name__ == '__main__':
    main()