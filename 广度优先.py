from collections import deque


def bfs(graph, start):
    visited = []  # 用于存储已访问过的节点
    queue = deque([start])  # 用队列存储待访问的节点
    visited.append(start)  # 将起始节点标记为已访问

    while queue:
        node = queue.popleft()  # 取出队列中的第一个节点
        # print(node)  # 可替换成对节点的其他操作

        # 遍历当前节点的邻居节点
        for neighbor in graph[node]:
            print(neighbor + "next")
            if neighbor not in visited:
                queue.append(neighbor)  # 将未访问过的邻居节点加入队列
                visited.append(neighbor)  # 将邻居节点标记为已访问


# 示例图的邻接表表示
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B"],
    "E": ["B", "F"],
    "F": ["C", "E"],
}

# 从节点 A 开始进行广度优先搜索
bfs(graph, "A")
