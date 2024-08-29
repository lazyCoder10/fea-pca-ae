import networkx as nx
import matplotlib.pyplot as plt

def create_graph_from_adjacency_list(adjacency_list):
    """ Create a graph from an adjacency list. """
    G = nx.Graph()
    for node, neighbors in enumerate(adjacency_list):
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    return G
#Max Threshold,0.3769927
#Factors,"[[2, 9], [3, 4, 6], [0, 8], [1, 5, 6, 7, 9], [1, 7], [3, 6], [1, 3, 5, 7, 9], [3, 4, 6, 9], [2], [0, 3, 6, 7]]"
#Max Threshold,0.4144748126255789
#Factors,"[[3, 7], [3, 4, 6], [4, 5], [0, 1, 4], [1, 2, 3, 6], [2], [1, 4], [0, 8], [7, 9], [8]]"
# Provided adjacency list
adjacency_list = [[3, 7], [3, 4, 6], [4, 5], [0, 1, 4], [1, 2, 3, 6], [2], [1, 4], [0, 8], [7, 9], [8]]

# Create the graph from the adjacency list
G = create_graph_from_adjacency_list(adjacency_list)

# Drawing the graph
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='blue', font_weight='bold')

# Saving the graph as a PNG file
plt.savefig("/Users/xuyingwangswift/Desktop/AE_FEA/AE/PCA_graph.png")

