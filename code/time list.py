import pickle
import matplotlib.pyplot as plt
import numpy as np

# 加载pickle文件
file_path = 'merged_time_series.pkl'
with open(file_path, 'rb') as f:
    time_series_data = pickle.load(f)

# 提取第四个时间序列
fourth_series = time_series_data[3]

# 检查第四个时间序列是否为二维数组
if isinstance(fourth_series, np.ndarray) and fourth_series.ndim == 2:
    # 计算振幅（这里假设振幅是行的绝对值之和）
    amplitude = np.sum(np.abs(fourth_series), axis=1)
    
    # 绘制第四个时间序列的振幅图
    plt.figure(figsize=(10, 6))
    plt.plot(amplitude)
    plt.title('Amplitude Plot of the Fourth Time Series')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
else:
    print("The fourth series is not a two-dimensional numpy array.")