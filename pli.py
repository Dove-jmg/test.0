import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle


def draw_rectangles(ax, start_y, color, num_rows=4):
    for i in range(num_rows):
        cbam_rect = Rectangle((3 + i * 4, start_y), 1, 1, edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(cbam_rect)


# 定义画布大小
fig, ax = plt.subplots(figsize=(8, 6))

# 创建图结构
G = nx.Graph()
nodes = [(i % 4, i // 4) for i in range(16)]
edges = [(i, (i + 1) % 16) for i in range(16)]
G.add_nodes_from(nodes)
G.add_edges_from(edges)
pos = {i: (j[0] * 1.5, j[1] * 1.5) for i, j in zip(range(16), nodes)}
nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue", edge_color="gray", ax=ax)

# 这里假设W的维度与节点数匹配
W = np.random.rand(len(G.nodes()), len(G.nodes()))
for u, v in G.edges():
    e = W[u, v]
    # 避免除零错误
    denominator = np.sum(e ** 2) ** 0.5
    if denominator == 0:
        denominator = 1e-8
    attention_score = e * W[v, u] * e / denominator
    color = attention_score / np.max(W)
    edge_color = plt.cm.viridis(color)[:3] / 255
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2.0, edge_color=edge_color, alpha=0.6, ax=ax)

for node in G.nodes():
    nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=300, node_color="skyblue", ax=ax)

colors = ['yellow', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'lime', 'teal', 'olive',
          'navy', 'maroon', 'salmon', 'gold', 'lavender', 'tan', 'coral', 'aquamarine', 'bisque', 'honeydew',
          'lavenderblush', 'mistyrose', 'azure', 'seagreen']
start_y_values = [-1] + list(range(4, 29))
for color, start_y in zip(colors, start_y_values):
    draw_rectangles(ax, start_y, color)

plt.show()
    