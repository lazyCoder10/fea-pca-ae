import numpy as np
import networkx as nx

def is_connected(G):
    """ Check if the graph is connected using DFS. """
    start_node = list(G.nodes())[0]
    visited = set()
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(set(G[node]) - visited)

    return len(visited) == len(G)

def find_largest_threshold_for_connected_graph(eigenvectors):
    """
    Find the largest threshold such that a connected graph can be built using all eigenvectors.

    Parameters:
    eigenvectors (list of np.array): List of eigenvectors.

    Returns:
    float: The largest threshold that allows building a connected graph.
    list of lists: List of indices of connected nodes for each node at this threshold.
    """
    n = len(eigenvectors[0])
    all_values = np.concatenate(eigenvectors)
    sorted_values = np.unique(np.sort(all_values))

    max_threshold = 0
    adjacency_list = []

    for threshold in sorted_values:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        for i in range(n):
            for j in range(i+1, n):
                if any(vec[i] > threshold and vec[j] > threshold for vec in eigenvectors):
                    G.add_edge(i, j)

        if is_connected(G):
            max_threshold = threshold
            adjacency_list = [list(G.neighbors(i)) for i in range(n)]
        else:
            break

    return max_threshold, adjacency_list

# Example usage with your provided eigenvectors
eigenvectors = [
    # Add your eigenvectors here as numpy arrays
]

threshold, adjacency_list = find_largest_threshold_for_connected_graph(eigenvectors)
print("Largest threshold for connected graph:", threshold)
print("Adjacency list at this threshold:", adjacency_list)

