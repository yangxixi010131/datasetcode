import pandas as pd
import numpy as np
import pickle

def calculate_pearson_correlation(time_series):
    # 将时间序列数组转换为 Pandas DataFrame
    df = pd.DataFrame(time_series)
    # 计算皮尔逊相关系数矩阵
    correlations = df.corr(method='pearson')
    return correlations

# 加载 pickle 文件中的时间序列数组
with open("merged_time_series.pkl", "rb") as f:
    time_series_new = pickle.load(f)

# 初始化一个空的列表，用于存储所有的相关系数矩阵
all_correlation_matrices = []

# 将 time_series_new 中的每个时间序列逐个提取，并计算相关系数矩阵
for i, time_series in enumerate(time_series_new):
    # print(f"Time Series {i+1}:")
    correlation_matrix = calculate_pearson_correlation(time_series)
    
    # print("Correlation Matrix:")
    # print(correlation_matrix)
    # print("\n-----------------------------------\n")
    
    # 将相关系数矩阵添加到列表中
    all_correlation_matrices.append(correlation_matrix.values)

# 合并所有相关系数矩阵为一个大的 numpy 数组
all_correlation_matrices_np = np.stack(all_correlation_matrices)

# 保存为一个单独的 numpy 文件
np.save('all_correlation_matrices103.npy', all_correlation_matrices_np)