import networkx as nx
import os
from matplotlib import pylab as plt

# 获取当前脚本文件所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 构建相对路径
file_name = "Chinanet.gml"
file_path = os.path.join(script_dir, file_name)

# 读取 GML 文件
G = nx.read_gml(file_path)

# 绘制图形（可选）
pos = nx.spring_layout(G)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=500,
    font_size=8,
    font_color="black",
    font_weight="bold",
)
plt.title("Graph from GML File")
plt.show()
