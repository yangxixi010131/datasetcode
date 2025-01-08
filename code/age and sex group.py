import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读取数据，指定没有列名
data = pd.read_csv('PPMI103.txt', sep='\t', header=None)

# 假设第二列是年龄，第三列是性别
# 定义年龄范围
age_bins = np.arange(10, 100, 10)
num_bins = len(age_bins) - 1  # 年龄段的数量

# 创建直方图
plt.figure(figsize=(12, 6))

# 初始化图例标签
male_label = 'Male'
female_label = 'Female'

# 为每个年龄段绘制直方图
for i in range(num_bins):
    # 筛选出当前年龄段的数据
    age_mask = (data[1] >= age_bins[i]) & (data[1] < age_bins[i+1])
    age_data = data[age_mask]
    
    # 为男性绘制直方图
    plt.hist(age_data[age_data[2] == 'M'][1], bins=[age_bins[i], age_bins[i+1]], 
             color='blue', alpha=0.5, label=male_label if i == 0 else "", 
             histtype='bar', rwidth=0.5, align='mid')

    # 为女性绘制直方图
    plt.hist(age_data[age_data[2] == 'F'][1], bins=[age_bins[i], age_bins[i+1]], 
             color='red', alpha=0.5, label=female_label if i == 0 else "", 
             histtype='bar', rwidth=0.5, align='mid')

# 添加图例
plt.legend(fontsize=20)

# 设置标题和坐标轴标签
plt.title('Age Distribution by Gender',fontsize=20)
plt.xlabel('Age',fontsize=20)
plt.ylabel('Number of People',fontsize=20)

# 设置X轴的刻度位置和标签
plt.xticks(age_bins[:-1] + 5, [f'{age_bins[i]}-{age_bins[i+1]-1}' for i in range(num_bins)],fontsize=20)
# 设置Y轴的刻度位置和标签
plt.yticks(fontsize=20)  # 设置Y轴刻度标签的大小
# 显示图形
plt.show()