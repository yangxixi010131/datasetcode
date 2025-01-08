import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import add_self_loops, remove_self_loops, to_scipy_sparse_matrix
from scipy.sparse import csgraph
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score

# 创建完全连接的图结构
def create_complete_edge_index(num_nodes):
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    
    edge_index = np.array(edge_index).T  # 转置以符合PyG的格式
    return torch.tensor(edge_index, dtype=torch.long)

# 计算拉普拉斯矩阵
def compute_laplacian(edge_index, num_nodes):
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    adj[edge_index[0], edge_index[1]] = 1
    adj = adj + adj.T
    adj = adj - torch.diag(adj.sum(dim=1))
    laplacian = adj
    return laplacian

# 加载数据
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
    
    edge_index = create_complete_edge_index(num_subjects)
    
    x = torch.tensor(matrices, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# 图卷积网络模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 训练和评估模型
def train(model, data, optimizer, laplacian):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)
    
    # 拉普拉斯正则化
    laplacian_loss = torch.trace(torch.matmul(out.T, torch.matmul(laplacian, out)))
    loss += 0.005 * laplacian_loss  # 正则化项的权重可以调整
    
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        
        # 获取真实标签
        true_labels = data.y.cpu().numpy()
        pred_labels = pred.cpu().numpy()
        
        # 计算准确率
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.num_nodes
        
        # 计算精确率和召回率
        precision = precision_score(true_labels, pred_labels, average='macro')
        recall = recall_score(true_labels, pred_labels, average='macro')

        return acc, precision, recall

def main():
    data = load_data()
    
    num_features = data.num_node_features
    hidden_channels = 64
    num_classes = len(torch.unique(data.y))
    
    model = GCN(num_features, hidden_channels, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 计算拉普拉斯矩阵
    laplacian = compute_laplacian(data.edge_index, data.num_nodes)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        loss = train(model, data, optimizer, laplacian)
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')
    
    # 最终评估
    final_acc, final_precision, final_recall = test(model, data)
    print(f'Final Accuracy: {final_acc:.4f}, Final Precision: {final_precision:.4f}, Final Recall: {final_recall:.4f}')

if __name__ == "__main__":
    main()