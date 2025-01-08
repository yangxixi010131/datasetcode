import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting, image

# 加载功能连接矩阵
correlation_matrices = np.load('all_correlation_matrices103.npy')

# 取出第一组患者的功能连接矩阵
first_subject_matrix = correlation_matrices[99]

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

# 绘制连接图
fig, ax = plt.subplots(figsize=(10, 10))

# 绘制连接图
plotting.plot_connectome(
    first_subject_matrix, 
    coords_mni,  # 使用 MNI 坐标
    title='Brain connectivity network', 
    edge_threshold='95%',  # 可以调整这个参数
    colorbar=True  # 添加颜色条以显示连接强度
)

# 显示大脑背景
display = plotting.plot_img(atlas_img, threshold=0, display_mode='ortho', cut_coords=(0, 0, 0), title='AAL Atlas', axes=ax)

# 添加标题
plt.savefig('connectome_subject_1.png')
plt.show()

# 画出第三张图，显示不同区域的标签，但不添加颜色条
fig, ax = plt.subplots(figsize=(10, 10))
plotting.plot_img(atlas_img, threshold=0, display_mode='ortho', cut_coords=(0, 0, 0), title='AAL Atlas with Labels', axes=ax)

# 添加区域标签
for i, label in enumerate(labels):
    ax.text(0, 0, i, label, fontsize=12, ha='right', va='center')

plt.savefig('aal_atlas_with_labels.png')
plt.show()  