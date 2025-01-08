import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
import matplotlib.pyplot as plt

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
        return x

    def laplacian_regularization(self, edge_index, num_nodes):
        # Compute Laplacian matrix
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
        laplacian = sp.csgraph.laplacian(adj, normed=True)
        laplacian = torch.tensor(laplacian.toarray(), dtype=torch.float32)
        return laplacian

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
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    
    laplacian = model.laplacian_regularization(data.edge_index, data.x.size(0)).to(device)
    
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        
        # Laplacian regularization for hidden layer output
        x = model.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=model.training)
        x = model.conv2(x, data.edge_index)
        x = F.relu(x)
        out_train = x[data.train_mask]
        laplacian_loss = torch.sum(torch.mm(out_train, torch.mm(laplacian[:out_train.size(1), :out_train.size(1)], out_train.t()))) / data.train_mask.sum().item()
        loss += 0.1 * laplacian_loss  # Adjust the lambda factor as needed
        
        loss.backward()
        optimizer.step()
        print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()))

def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out[data.test_mask].argmax(dim=1)
        correct = int(pred.eq(data.y[data.test_mask]).sum().item())
        acc = correct / int(data.test_mask.sum())
        print('GCN Accuracy: {:.4f}'.format(acc))
        
        # 返回预测和实际标签
        return pred.cpu().numpy(), data.y[data.test_mask].cpu().numpy(), data.test_mask.cpu().numpy()

def visualize_predictions(predictions, true_labels, data):
    # 假设节点特征为二维
    x = data.x.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    
    # 仅选择测试节点
    x_test = x[test_mask]
    predictions_test = predictions
    true_labels_test = true_labels
    
    plt.figure(figsize=(8, 6))
    
    # 绘制预测的类别
    plt.scatter(x_test[:, 0], x_test[:, 1], c=predictions_test, cmap='coolwarm', marker='o', s=50, edgecolors='k', alpha=0.7)
    
    # 添加图例和标题
    plt.colorbar(label='Predicted Class')
    plt.title('Node Classification Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.show()

def main():
    x, edge_index, y, num_features = load_data()
    data = create_data(x, edge_index, y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_node_features=num_features).to(device)
    
    data = data.to(device)
    
    train(model, data, device)
    predictions, true_labels, test_mask = test(model, data)
    
    visualize_predictions(predictions, true_labels, data)

if __name__ == '__main__':
    main()