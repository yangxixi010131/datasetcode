import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)  # 增加隐藏层神经元数目
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 2)  # 二分类
        self.norm = torch.nn.BatchNorm1d(128)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# 数据加载函数
def load_data():
    # 加载功能连接矩阵
    matrices = np.load('all_correlation_matrices103.npy')
    
    # 加载标签信息
    labels_df = pd.read_csv('PPMI103.txt', sep='\t', header=None)
    labels = labels_df[3].apply(lambda x: 0 if x == 'PD' else 1).values
    
    # 处理缺失值
    if np.isnan(matrices).any():
        matrices = np.nan_to_num(matrices)

    if pd.isna(labels).any():
        labels = np.nan_to_num(labels)
    
    num_subjects = matrices.shape[0]
    num_features = matrices.shape[1] * matrices.shape[2]

    # 特征标准化
    scaler = StandardScaler()
    matrices = matrices.reshape(num_subjects, -1)
    matrices = scaler.fit_transform(matrices)
    
    # 载入新的邻接矩阵
    similarity_matrix = np.load('fused_similarity_matrix_k20.npy')
    
    # 生成边索引
    edge_index = np.array(np.nonzero(similarity_matrix > 0.5))
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    # 特征矩阵
    x = torch.tensor(matrices, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)  # 标签为长整型
    
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
    loss_function = torch.nn.CrossEntropyLoss().to(device)  # 使用交叉熵损失
    model.train()
    for epoch in range(100):  # 训练100个epoch
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        # 每个epoch打印一次
        print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()))

# 测试函数
def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out[data.test_mask].argmax(dim=1)  # 获取预测标签

        # 获取真实标签
        true_labels = data.y[data.test_mask].cpu().numpy()
        pred_labels = pred.cpu().numpy()

        # 计算准确率
        correct = int(pred.eq(data.y[data.test_mask]).sum().item())
        acc = correct / int(data.test_mask.sum())

        # 计算精确率和召回率
        precision = precision_score(true_labels, pred_labels, average='macro')
        recall = recall_score(true_labels, pred_labels, average='macro')

        print(f'GCN Accuracy: {acc:.4f}')
        print(f'GCN Precision: {precision:.4f}')
        print(f'GCN Recall: {recall:.4f}')

# 主函数
def main():
    x, edge_index, y, num_features = load_data()
    data = create_data(x, edge_index, y)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(num_node_features=num_features).to(device)
    
    data = data.to(device)
    
    train(model, data, device)
    test(model, data)

if __name__ == '__main__':
    main()