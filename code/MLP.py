import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 数据加载
# 加载功能连接矩阵
feature_matrices = np.load('all_correlation_matrices103.npy')

# 加载标签数据
labels_df = pd.read_csv('PPMI103.txt', sep='\t', header=None)
labels_df = labels_df[[0, 3]]  # 选择编号和标签信息
labels_df.columns = ['id', 'label']

# 将标签转换为0和1
labels_df['label'] = labels_df['label'].apply(lambda x: 0 if x == 'PD' else 1)

# 根据编号对齐标签和功能连接矩阵
labels_dict = dict(zip(labels_df['id'], labels_df['label']))
max_id = max(labels_dict.keys())
labels = np.array([labels_dict.get(i, np.nan) for i in range(max_id + 1)])  # 编号从0开始
features = feature_matrices

# 2. 数据预处理
# 标准化特征
scaler = StandardScaler()
features = np.array([scaler.fit_transform(fm) for fm in features])

# 检查数据中是否有 NaN 或 Inf 值，并处理
def clean_data(features, labels):
    # 检查并处理 features 中的 NaN 或 Inf 值
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 0
    
    # 处理 labels 中的 NaN 值
    valid_indices = ~np.isnan(labels)
    features = features[valid_indices]
    labels = labels[valid_indices]
    
    return features, labels

features, labels = clean_data(features, labels)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 将数据转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 3. 模型构建
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(116*116, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 2)  # 输出层：2个类

        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, 116*116)  # 扁平化
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # 调整学习率

# 4. 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    # 前向传播
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    # 打印每一代的损失值
    print(f'Epoch {epoch + 1:03d} loss {loss.item():.4f}')
    
    # 检查损失值是否为 NaN
    if torch.isnan(loss):
        print("训练过程中损失值为 NaN，停止训练")
        break

# 5. 模型评估
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    
    accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
    precision = precision_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')
    recall = recall_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')