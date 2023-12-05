import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque

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
node_attributes = {}
# 添加节点属性
# {node:{neibor1:a,neibor2:b,neibor3:c}}
# 获取节点属性nx.get_node_attributes(G, "neibor")["node"]
for node in nodes:
    node_attributes[node] = {"count": 0}
    for neighbor in G[node]:
        # 统计邻居节点被攻破数量/概率
        node_attributes[node][neighbor] = 0.0
nx.set_node_attributes(G, node_attributes)


def send_msg_label(G, from_, to_):
    msg = 1
    to_node = G.nodes[to_]
    to_node["count"] = to_node["count"] - to_node[from_]  # 减去原来的节点信息
    to_node["count"] = to_node["count"] + msg  # 加上新的msg
    # 更新对应msg
    to_node[from_] = msg


# 隐藏节点向其他隐藏节点发送信息
def send_msg(G, from_, to_):
    lamuda = 0.5  # 参数大小影响传播概率
    count = G.nodes[from_]["count"] - G.nodes[from_][to_]
    if (G.nodes[from_]["count"] < 0) | (G.nodes[to_]["count"] < 0):
        print("<0error,1")
    nei_sum = len(G[from_])
    p = count / (nei_sum - 1 + lamuda)
    msg = p
    to_node = G.nodes[to_]
    to_node["count"] = to_node["count"] - to_node[from_]  # 减去原来的节点信息
    to_node["count"] = to_node["count"] + msg  # 加上新的msg
    if (G.nodes[from_]["count"] < 0) | (G.nodes[to_]["count"] < 0):
        print("<0error,1")
    # 更新对应msg
    to_node[from_] = msg


def step(G, node_a, node_d):
    node_a1 = []
    # 从带有标签（被攻破）的节点开始传播node_a
    if type(node_a[0]) != str:  # 攻击者使用路由追踪策略时，节点集可能不止一个
        for listnode in node_a:
            for i in listnode:
                node_a1.append(i)
    else:
        node_a1 = node_a
    for node in node_a1:
        if node not in node_d:  # 被攻破表明不在防守范围内
            for neighbor in G[node]:
                if (neighbor not in node_d) & (neighbor not in node_a1):  # 被保护的节点不受影响
                    send_msg_label(G, node, neighbor)

    # 从不带标签的节点开始传播
    for node in list(G.nodes()):
        if (node not in node_a1) & (node not in node_d):
            for neighbor in G[node]:
                if (neighbor not in node_d) & (neighbor not in node_a1):
                    send_msg(G, node, neighbor)


node_a = [
    ["Server1", "Router3", "Switch5", "PC6"],
    ["PC5", "Switch4", "Router6", "Server3"],
    ["Server2", "Router5", "Router6", "Switch4", "PC5"],
]
edge_a = [
    ("Server1", "Router3"),
    ("Router3", "Switch5"),
    ("Switch5", "PC6"),
    ("PC5", "Switch4"),
    ("Switch4", "Router6"),
    ("Router6", "Server3"),
    ("Server2", "Router5"),
    ("Router5", "Router6"),
    ("Router6", "Switch4"),
    ("Switch4", "PC5"),
]

node_d = ["Router5"]

for i in range(1):
    step(G, node_a, node_d)

result = []

for i in nodes:
    if (i not in node_a) & (i not in node_d):
        a = G.nodes[i]["count"]
        result.append(a)

# print(result)
print(G["Router5"]["Server2"])
