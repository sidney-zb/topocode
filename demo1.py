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
# 添加节点属性
# {node:{neibor1:a,neibor2:b,neibor3:c}}
# 获取节点属性nx.get_node_attributes(G, "neibor")["node"]
# 修改属性nx.set_node_attributes(G, {"node": {"neibor": "x"}})
for node in nodes:
    node_attributes[node] = {"count": 0}
    for neighbor in G[node]:
        # 统计邻居节点被攻破数量/概率
        node_attributes[node][neighbor] = 0.0
nx.set_node_attributes(G, node_attributes)


for edge in edges:
    source, target, weight = edge
    i = nodes.index(source)
    j = nodes.index(target)
    adjacency_matrix[i, j] = weight
    adjacency_matrix[j, i] = weight  # 无向图需要对称矩阵
# 打印邻接矩阵
# print(adjacency_matrix)

# 计算节点BC值
betweenness_centrality = nx.betweenness_centrality(G)
# print(betweenness_centrality)

# BC中值排序
sorted_data = dict(
    sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)
)
# print(sorted_data)


# 防御方节点保护算法
def node_protection(G):
    # 可用防御资源总数
    D_Source = 0.5
    node_name = []
    node_BC = []
    # 被保护的节点
    node_pro = []
    # 被保护得边集合
    link1 = []
    # 计算每个节点的中介中心性
    betweenness_centrality = nx.betweenness_centrality(G)
    # 排序
    sorted_data = dict(
        sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)
    )
    for key, value in sorted_data.items():
        node_name.append(key)
        node_BC.append(value)

    tempBC_sum = 0
    # 节点混淆
    for i in range(0, len(node_name)):
        if tempBC_sum >= D_Source:
            break
        node_pro.append(node_name[i])
        for neighbor in G[node_name[i]]:
            link1.append((node_name[i], neighbor))
            link1.append((neighbor, node_name[i]))  # 处理无向边问题
        tempBC_sum = tempBC_sum + node_BC[i]
    return node_pro, link1


# 得到被保护的节点
node_d, edge_d = node_protection(G)


# 防御方随机节点保护策略
def node_pro_random(G):
    # 可用防御资源总数
    D_Source = 3
    node_name = []
    node_BC = []
    # 被保护的节点
    node_pro = []
    # 被保护得边集合
    link4 = []
    # 计算每个节点的中介中心性
    betweenness_centrality = nx.betweenness_centrality(G)

    sorted_data = dict(
        sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)
    )
    for key, value in sorted_data.items():
        node_name.append(key)
        node_BC.append(value)

    tempBC_sum = 0
    # 随机选择节点混淆
    while tempBC_sum <= D_Source:
        target_pro = random.choice(node_name)
        if target_pro not in node_pro:
            node_pro.append(target_pro)
            tempBC_sum = tempBC_sum + betweenness_centrality[target_pro]

    for i in node_pro:
        for neighbor in G[i]:
            link4.append((i, neighbor))
            link4.append((neighbor, i))

    return node_pro, link4


# node_d, edge_d = node_pro_random(G)


# 攻击者策略：端到端探测
def attack_p2p(G):
    A_Source = 3
    node_name = []
    node_BC = []
    # 候选节点集
    choice_node1 = []
    # 选取路径
    result_path = []
    # 计算每个节点的中介中心性
    betweenness_centrality = nx.betweenness_centrality(G)
    # 排序
    sorted_data = dict(
        sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)
    )

    for key, value in sorted_data.items():
        node_name.append(key)
        node_BC.append(value)
    # 随机选择一个BC值为0的节点进行探测
    for i in range(0, len(node_name)):
        if node_BC[i] == 0.0:
            choice_node1.append(node_name[i])
    # 选取某路径消耗的资源
    total_bc = 0
    shortest_path_edges = []
    for t in range(0, 1000):
        while total_bc < A_Source:
            # 随机选择起始节点和终止节点
            start_node = random.choice(choice_node1)
            end_node = random.choice(choice_node1)
            # 除去初始和终止节点相同的情况
            if start_node == end_node:
                continue
            # 最短路径的上节点
            shortest_path = nx.shortest_path(
                G, source=start_node, target=end_node, weight="weight"
            )
            for j in shortest_path:
                total_bc = sorted_data[j] + total_bc
            # 最短路径的边
            if total_bc < A_Source:
                A_Source = A_Source - total_bc
                result_path.append(shortest_path)
                # 处理无向边问题
                for i in range(len(shortest_path) - 1):
                    shortest_path_edges.append((shortest_path[i], shortest_path[i + 1]))
            else:
                total_bc = 0
    return result_path, shortest_path_edges


# node_a, edge_a = attack_p2p(G)


# 广度优先算法
def BFS(G, source):
    node_name = []
    node_BC = []
    # 候选节点集
    choice_node2 = []
    # 把边也打印出来
    link3 = []
    betweenness_centrality = nx.betweenness_centrality(G)
    # 排序
    sorted_data = dict(
        sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)
    )
    for key, value in sorted_data.items():
        node_name.append(key)
        node_BC.append(value)
    # 随机选择一个BC值为0的节点进行广度优先
    for i in range(0, len(node_name)):
        if node_BC[i] == 0.0:
            choice_node2.append(node_name[i])
    start_node = random.choice(choice_node2)

    visited = []  # 用于存储已访问过的节点
    queue = deque([start_node])  # 用队列存储待访问的节点
    visited.append(start_node)  # 将起始节点标记为已访问
    tag = 0
    while queue:
        total_BC = 0
        # 计算当前已用资源
        for node_check in visited:
            total_BC = total_BC + sorted_data[node_check]
        if (total_BC >= source) | (tag > 0):
            break
        node = queue.popleft()  # 取出队列中的第一个节点
        # 遍历当前节点的邻居节点
        for neighbor in G[node]:
            total_BC = total_BC + sorted_data[neighbor]
            if total_BC >= source:
                tag = 1
                break
            if neighbor not in visited:
                queue.append(neighbor)  # 将未访问过的邻居节点加入队列
                visited.append(neighbor)  # 将邻居节点标记为已访问
                link3.append((node, neighbor))

    return visited, link3


# 攻击者策略：路由追踪
def attack_traceout(G):
    A_Source = 2
    # 得到节点集合
    result2 = BFS(G, A_Source)
    return result2


node_a, edge_a = attack_traceout(G)

# {node:{neibor1:a,neibor2:b,neibor3:c}}
# 获取节点属性nx.get_node_attributes(G, "neibor")["node"]
# 获取节点属性G[node][neibor1]
# 修改属性nx.set_node_attributes(G, {"node": {"neibor": "x"}})
# 被探测成功的节点想隐藏节点发送信息
# G.nodes[node] 返回节点的属性字典。
# G[node] 返回邻接节点的视图(返回邻居节点)。


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


# 计算双方收益（零和）
def payoff_calculation(G, node_a, edge_a, node_d, edge_d, matrix):
    # 计算攻击者收益
    n_payoff_t = 0
    n_payoff_f = 0
    e_payoff_t = 0
    e_payoff_f = 0
    betweenness_centrality = nx.betweenness_centrality(G)

    # 攻击者使用路由追踪策略时，节点集可能不止一个
    print(node_a)
    # print(edge_a)
    print(node_d)
    if type(node_a[0]) == str:
        for j in node_a:
            if j not in node_d:
                n_payoff_t = n_payoff_t + betweenness_centrality[j]
            # 计算节点负收益
            else:
                n_payoff_f = n_payoff_f + betweenness_centrality[j]
    else:
        for i in range(len(node_a)):
            for node in node_a[i]:
                # 计算节点正收益
                if node not in node_d:
                    n_payoff_t = n_payoff_t + betweenness_centrality[node]
                # 计算节点负收益
                else:
                    n_payoff_f = n_payoff_f + betweenness_centrality[node]
    node_list = list(G.nodes)
    for edge in edge_a:
        source = node_list.index(edge[0])
        target = node_list.index(edge[1])
        # 计算边正收益
        if edge not in edge_d:
            e_payoff_t = e_payoff_t + matrix[source, target]
        # 计算边负收益
        else:
            e_payoff_f = e_payoff_f + matrix[source, target]

    # 消息传播
    for i in range(10):
        step(G, node_a, node_d)

    lbp = 0
    for i in list(G.nodes()):
        if (i not in node_a) & (i not in node_d):
            a = G.nodes[i]["count"]
            a = round(a, 4)
            sum = len(list(G.neighbors(i)))
            p = a / sum
            p = round(p, 3)
            lbp = p + lbp
    return n_payoff_t - n_payoff_f + 0.1 * (e_payoff_t - e_payoff_f) + 0.2 * lbp


result = payoff_calculation(G, node_a, edge_a, node_d, edge_d, adjacency_matrix)
print("攻击者收益为：", result)

for i in nodes:
    if (i not in node_a) & (i not in node_d):
        a = G.nodes[i]["count"]
        # print(f"{i}:{a}")

# 绘制带权图
"""
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
plt.title("带权网络拓扑结构图")
# plt.show()
"""
