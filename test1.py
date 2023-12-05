import networkx as nx
import matplotlib.pyplot as plt

# 创建一个带权图
G = nx.Graph()

# 添加节点
nodes = ["Router", "Server1", "Server2", "Switch1", "Switch2", "PC1", "PC2", "PC3"]
G.add_nodes_from(nodes)

# 添加带权边
edges = [
    ("Router", "Server1", {"weight": 2}),  # 为这条边添加权重为2
    ("Router", "Server2", {"weight": 3}),  # 为这条边添加权重为3
    ("Router", "Switch1", {"weight": 1}),  # 为这条边添加权重为1
    ("Switch1", "PC1", {"weight": 4}),  # 为这条边添加权重为4
    ("Switch1", "PC2", {"weight": 2}),  # 为这条边添加权重为2
    ("Switch2", "PC3", {"weight": 5}),  # 为这条边添加权重为5
]
G.add_edges_from(edges)

# 绘制图形
pos = nx.spring_layout(G)  # 布局算法
edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1000,
    node_color="skyblue",
    font_size=10,
    font_color="black",
    font_weight="bold",
)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
plt.title("单位内网拓扑结构图（带权边）")
plt.show()
