import networkx as nx
import matplotlib.pyplot as plt

# 创建一个无向图
G = nx.Graph()

# 添加节点
nodes = [
    "Router1",
    "Router2",
    "Server1",
    "Server2",
    "Switch1",
    "Switch2",
    "PC1",
    "PC2",
    "PC3",
    "Printer1",
]
G.add_nodes_from(nodes)

# 添加边
edges = [
    ("Router1", "Server1", {"weight": 3}),
    ("Router1", "Switch1", {"weight": 2}),
    ("Router2", "Server2", {"weight": 4}),
    ("Router2", "Switch2", {"weight": 2}),
    ("Switch1", "PC1", {"weight": 1}),
    ("Switch1", "PC2", {"weight": 1}),
    ("Switch2", "PC3", {"weight": 2}),
    ("Switch2", "Printer1", {"weight": 1}),
    ("Router1", "Router2", {"weight": 4}),
]
G.add_edges_from(edges)

# 计算中介中心性
betweenness_centrality = nx.betweenness_centrality(G)
print(betweenness_centrality)
# 打印每个节点的中介中心性
# for node, centrality in betweenness_centrality.items():
# print(f"节点 {node} 的中介中心性为 {centrality:.2f}")
# 绘制图形


# 防御方节点保护算法
def node_protection(G):
    # 计算每个节点的中介中心性
    betweenness_centrality = nx.betweenness_centrality(G)
    sorted_data = dict(
        sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)
    )


pos = nx.spring_layout(G, seed=42)  # 布局算法
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
