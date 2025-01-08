import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# 加载功能连接矩阵数据
all_correlation_matrices_np = np.load('all_correlation_matrices103.npy', allow_pickle=True)
num_subjects = all_correlation_matrices_np.shape[0]

# 创建无向图
G = nx.Graph()

# 添加节点
G.add_nodes_from(range(num_subjects))

# 计算相似性并添加边
for i in range(num_subjects):
    for j in range(i + 1, num_subjects):
        # 计算皮尔逊相关系数的平均值作为相似性度量
        correlation = np.mean(all_correlation_matrices_np[i, j])
        if correlation > 0.0:  # 仅添加非零相关性的边
            G.add_edge(i, j, weight=correlation)

# 保存图对象为Pickle文件
with open('functional_connectivity_graph.pkl', 'wb') as f:
    pickle.dump(G, f)

# 获取边的权重（即相关性）
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

# 使用Spring布局算法布置节点，seed参数确保结果可重现
pos = nx.spring_layout(G, seed=42)

# 绘制图形
plt.figure(figsize=(18, 15))

# 绘制节点，调整节点大小和透明度
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500, alpha=0.8)

# 绘制节点标签
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_family='sans-serif', labels={node: node for node in G.nodes()})

# 调整颜色映射范围，使颜色更深更浅
min_weight = min(edge_weights)
max_weight = max(edge_weights)

# 绘制边，根据权重设置颜色和宽度
edges = nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color=edge_weights, edge_cmap=plt.cm.Blues, edge_vmin=min_weight, edge_vmax=max_weight)

# 添加颜色条，并指定放置的坐标轴
cbar = plt.colorbar(edges, ax=plt.gca())
cbar.ax.tick_params(labelsize=20)  # 设置颜色条刻度标签的大小
plt.title('Subject - subject similarity network',fontsize=20)
plt.axis('off')
plt.show()       