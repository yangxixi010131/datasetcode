import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, image

# 加载功能连接矩阵
correlation_matrices = np.load('all_correlation_matrices103.npy')

# 取出第一组患者的功能连接矩阵
first_subject_matrix = correlation_matrices[0]

# 加载AAL脑图谱模板
atlas = datasets.fetch_atlas_aal()
atlas_img = image.load_img(atlas.maps)  # 加载脑图谱图像

# 获取AAL模板的标签和坐标
labels = atlas.labels
atlas_data = atlas_img.get_fdata()

# 获取 AAL 脑图谱的空间坐标
coords = np.array(np.nonzero(atlas_data)).T  # 获取非零元素的坐标

# 获取每个区域的中心坐标
unique_labels = np.unique(atlas_data[coords[:, 0], coords[:, 1], coords[:, 2]])
coords_unique = []
for label in unique_labels:
    # 获取该标签的所有坐标
    label_coords = coords[atlas_data[coords[:, 0], coords[:, 1], coords[:, 2]] == label]
    # 计算均值坐标
    mean_coord = label_coords.mean(axis=0)
    coords_unique.append(mean_coord)

# 将坐标转换为 numpy 数组并确保其形状正确
coords = np.array(coords_unique)

# 转换为 MNI 空间
affine = atlas_img.affine  # 获取 AAL 脑图谱的仿射矩阵
coords_mni = np.dot(coords, affine[:3, :3].T) + affine[:3, 3]  # 应用仿射变换

# 定义脑叶分组
brain_regions = {
    "Frontal": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33],
    "Occipital": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    "Parietal": [34, 35, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
    "Subcortical": [28, 29, 70, 71, 72, 73, 74, 75, 76, 77],
    "Temporal": [36, 37, 38, 39, 40, 41, 54, 55, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
    "Cerebellum": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
}

# 创建一个2x3的子图布局
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# 定义脑叶分组的标题
region_titles = ["Frontal", "Occipital", "Parietal", "Subcortical", "Temporal", "Cerebellum"]

# 迭代每个脑叶并绘制连接图
for i, (region_name, region_indices) in enumerate(brain_regions.items()):
    # 创建只包含特定脑叶区域的连接矩阵
    region_matrix = first_subject_matrix[np.ix_(region_indices, region_indices)]
    
    # 绘制连接图
    plotting.plot_connectome(
        region_matrix, 
        coords_mni[region_indices],  # 使用该脑叶的MNI坐标
        edge_threshold='95%',  # 可以调整这个参数
        colorbar=False,  # 不添加颜色条以显示连接强度
        display_mode='ortho',  # 显示矢状、轴向和冠状视图
        black_bg=False,  # 使用白色背景
        axes=axs.flat[i]  # 将图像绘制在相应的子图上
    )
    
    # 使用fig.text方法手动添加居中的标题
    # 计算每个子图上方标题的位置
    pos = axs.flat[i].get_position()
    x = 0.5 * (pos.x0 + pos.x1)
    y = pos.y1 + 0.05  # 在子图上方留出一些空间
    fig.text(x, y, region_titles[i], fontsize=20, weight='bold', ha='center', va='center')

# 调整子图间距
plt.tight_layout()
plt.show()