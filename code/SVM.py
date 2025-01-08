import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer

# 加载功能连接矩阵数据
correlation_matrices = np.load('all_correlation_matrices103.npy')

# 加载标签数据
labels_df = pd.read_csv('PPMI103.txt', delimiter='\t', header=None)
labels_df = labels_df.iloc[:, [0, 3]]  # 选择编号和标签列
labels_df.columns = ['ID', 'Label']

# 将标签信息转化为0和1
labels_df['Label'] = labels_df['Label'].apply(lambda x: 0 if x == 'PD' else 1)

# 确保矩阵和标签的顺序一致
if len(correlation_matrices) != len(labels_df):
    raise ValueError("The number of correlation matrices does not match the number of labels.")

# 合并功能连接矩阵和标签信息
labels_df['Matrix'] = list(correlation_matrices)

# 提取特征和标签
X = np.array(labels_df['Matrix'].tolist())
y = labels_df['Label'].values

# 检查是否有NaN值
if np.any(np.isnan(X)):
    print("NaN values detected in the feature matrix. Imputing missing values.")
    
    # 使用SimpleImputer进行缺失值填充
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据扁平化为二维数组
n_samples, n_features1, n_features2 = X_train.shape
X_train_flat = X_train.reshape(n_samples, n_features1 * n_features2)
n_samples, n_features1, n_features2 = X_test.shape
X_test_flat = X_test.reshape(n_samples, n_features1 * n_features2)

# 标准化特征
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# 创建SVM模型
svm_model = SVC(kernel='linear', random_state=42)

# 训练模型
svm_model.fit(X_train_flat, y_train)

# 预测
y_pred = svm_model.predict(X_test_flat)

# 计算准确率、精确率和召回率
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')

# 打印结果
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')