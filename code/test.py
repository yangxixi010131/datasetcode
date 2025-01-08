import numpy as np

# 加载.npy文件
matrix = np.load('2_all_subjects_similarity_matrices.npy')

# 打印矩阵的形状
print(matrix.shape)
# import pickle

# # 加载.pkl文件
# with open('time_series_test_new.pkl', 'rb') as file:
#     data = pickle.load(file)

# # 打印数据类型
# print(type(data))
# # 如果数据是列表
# if isinstance(data, list):
#     print("Data is a list with length:", len(data))
#     # 打印列表的前几个元素
#     print("First few elements:", data[:5])

# # 如果数据是字典
# elif isinstance(data, dict):
#     print("Data is a dictionary with keys:", list(data.keys()))
#     # 打印字典的前几个键值对
#     print("First few key-value pairs:", list(data.items())[:5])