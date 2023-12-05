dict = {
    "Router2": 0.66,
    "Router4": 0.5733333333333334,
    "Router5": 0.4766666666666667,
    "Router3": 0.4166666666666667,
    "Router6": 0.4166666666666667,
    "Router1": 0.35333333333333333,
    "Switch2": 0.15666666666666668,
    "Switch6": 0.15666666666666668,
    "Switch7": 0.15666666666666668,
    "Switch1": 0.08,
    "Switch3": 0.08,
    "Switch4": 0.08,
    "Switch5": 0.08,
    "Server1": 0.0,
    "Server2": 0.0,
    "Server3": 0.0,
    "PC1": 0.0,
    "PC2": 0.0,
    "PC3": 0.0,
    "PC4": 0.0,
    "PC5": 0.0,
    "PC6": 0.0,
    "PC7": 0.0,
    "PC8": 0.0,
    "PC9": 0.0,
    "PC10": 0.0,
}
list1 = []
list2 = []
for key, value in dict.items():
    list1.append(key)
    list2.append(value)

# print(list1)
# print(list2)
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 创建一个带权图
G = nx.Graph()

# 节点列表
nodes = [
    "Router1",
    "Router2",
    "Router3",
    "Router4",
    "Router5",
    "Router6",
    "Server1",
    "Server2",
    "Server3",
    "Switch1",
    "Switch2",
    "Switch3",
    "Switch4",
    "Switch5",
    "Switch6",
    "Switch7",
    "PC1",
    "PC2",
    "PC3",
    "PC4",
    "PC5",
    "PC6",
    "PC7",
    "PC8",
    "PC9",
    "PC10",
]

G.add_nodes_from(nodes)
# 创建邻接矩阵
num_nodes = len(nodes)
adjacency_matrix = np.zeros((num_nodes, num_nodes))  # 初始化为零矩阵

# 添加带权边到图
edges = [
    ("Router1", "Switch1", 2),
    ("Router1", "Switch2", 2),
    ("Router1", "Router2", 3),
    ("Router2", "Router3", 4),
    ("Router2", "Router4", 4),
    ("Router3", "Switch5", 2),
    ("Router3", "Switch6", 3),
    ("Router3", "Server1", 1),
    ("Router4", "Router5", 5),
    ("Router4", "Switch3", 2),
    ("Router5", "Server2", 2),
    ("Router5", "Router6", 3),
    ("Router6", "Switch4", 2),
    ("Router6", "Switch7", 3),
    ("Router6", "Server3", 2),
    ("Switch1", "PC1", 1),
    ("Switch2", "PC2", 1),
    ("Switch2", "PC3", 1),
    ("Switch3", "PC4", 1),
    ("Switch4", "PC5", 1),
    ("Switch5", "PC6", 1),
    ("Switch6", "PC7", 1),
    ("Switch6", "PC8", 1),
    ("Switch7", "PC9", 1),
    ("Switch7", "PC10", 1),
]

G.add_weighted_edges_from(edges)  # 使用add_weighted_edges_from添加带权边

print(G.edges())

start_node = "PC1"
end_node = "PC2"

print(G["Router1"]["Switch1"]["weight"])

shortest_path = nx.shortest_path(G, source=start_node, target=end_node, weight="weight")
# 输出最短路径的节点和边
print("最短路径节点序列：", shortest_path)

# 通过节点序列获取最短路径上的边
shortest_path_edges = [
    (shortest_path[i], shortest_path[i + 1]) for i in range(len(shortest_path) - 1)
]

print("最短路径边序列：", shortest_path_edges)

# 遍历边并打印其权重
total_weight = 0
for i in range(len(shortest_path) - 1):
    source_node = shortest_path[i]
    target_node = shortest_path[i + 1]
    edge_weight = G[source_node][target_node]["weight"]
    total_weight += edge_weight
    print(f"边 ({source_node}, {target_node}) 的权重：{edge_weight}")

print("最短路径的总权重：", total_weight)

# print(G.nodes())
