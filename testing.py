import networkx as nx
import random

# 创建一个无向图
G = nx.Graph()

# 添加真实节点
real_nodes = ["Node1", "Node2", "Node3", "Node4", "Node5"]
G.add_nodes_from(real_nodes)

# 添加边
edges = [
    ("Node1", "Node2"),
    ("Node2", "Node3"),
    ("Node2", "Node4"),
    ("Node3", "Node5"),
    ("Node4", "Node5"),
]
G.add_edges_from(edges)

# 添加虚假节点
fake_nodes = ["FakeNode1", "FakeNode2"]
G.add_nodes_from(fake_nodes)

# 引入虚假边
fake_edges = [
    ("Node1", "FakeNode1"),
    ("Node4", "FakeNode2"),
]
G.add_edges_from(fake_edges)


# 生成虚假流量
def generate_fake_traffic(graph, source, target):
    # 模拟虚假流量生成
    fake_traffic = random.randint(1, 10)
    print(f"生成虚假流量：从 {source} 到 {target}，流量量：{fake_traffic}")


# 模拟真实流量
generate_fake_traffic(G, "Node1", "Node3")
generate_fake_traffic(G, "Node2", "Node5")

# 模拟虚假流量
generate_fake_traffic(G, "Node1", "FakeNode1")
generate_fake_traffic(G, "Node4", "FakeNode2")

# 绘制图形
import matplotlib.pyplot as plt

pos = nx.spring_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=500,
    node_color="lightblue",
    font_size=10,
    font_color="black",
    font_weight="bold",
)
plt.title("网络拓扑图")
plt.show()
