import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from FactorArchitecture import FactorArchitecture
from Function import Function
from FEA import FEA
from pso import PSO
import csv
import os


def write_to_file(path_dir, csv_file_name, max_threshold, factors):
    # Ensure the directory exists or create it if necessary
    os.makedirs(path_dir, exist_ok=True)

    # Create a list with the data you want to write to the CSV file
    data_to_write = [("Max Threshold", max_threshold), ("Factors", factors)]

    file_path = os.path.join(path_dir, csv_file_name)
    # Open the CSV file for writing
    with open(file_path, 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the data to the CSV file
        csv_writer.writerows(data_to_write)

    print(f"Data has been written to {file_path}")


def load_and_prepare_pca_data(dataset):
    """
    Load and preprocess data.
    """
    #df = pd.read_csv(file_path)

    # Scale the data
    scaler = MinMaxScaler()
    X_std = scaler.fit_transform(dataset)
    return X_std


##########################################

def get_eigenvectors(X_std):
    """
    Perform PCA and extract factors.
    """
    pca_full = PCA()
    pca_full.fit(X_std)

    # Extracting the eigenvectors
    eigenvectors = pca_full.components_
    eigenvectors_without_last_feature = np.array([vector[:-1] for vector in eigenvectors])
    return eigenvectors_without_last_feature


###########################################

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


#########################################

def find_largest_threshold_for_connected_graph(eigenvectors):
    """
    Find the largest threshold such that a connected graph can be built using all eigenvectors.

    Parameters:
    eigenvectors (np.ndarray): A 2D numpy array where each row is an eigenvector.

    Returns:
    float: The largest threshold that allows building a connected graph.
    list of lists: List of indices of connected nodes for each node at this threshold.
    """
    n = eigenvectors.shape[1]  # Number of features (columns)
    all_values = eigenvectors.flatten()
    sorted_values = np.unique(np.sort(all_values))

    max_threshold = 0
    adjacency_list = []

    for threshold in sorted_values:
        G = nx.Graph()
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                # Use element-wise logical 'and' operation
                if any((eigenvectors[:, i] > threshold) & (eigenvectors[:, j] > threshold)):
                    G.add_edge(i, j)

        if is_connected(G):
            max_threshold = threshold
            adjacency_list = [list(G.neighbors(i)) for i in range(n)]
        else:
            break

    print("Largest Threshold for Connected Graph:", max_threshold)
    print("Adjacency List at this Threshold:", adjacency_list)

    return max_threshold, adjacency_list


###############################################################


def run_pca_fea_process(dataset,
                        num_factors,
                        fea_runs,
                        generations,
                        pop_size,
                        fcn_num, lb, ub,
                        base_performance_result_dir,
                        performance_result_file,
                        threshold_dir,
                        max_threshold_factors_file):
    """
    Function to setup and run the Factored Evolutionary Algorithm process.
    """
    X_std = load_and_prepare_pca_data(dataset)
    eigenvectors = get_eigenvectors(X_std)
    max_threshold, factors = find_largest_threshold_for_connected_graph(eigenvectors)
    write_to_file(threshold_dir, max_threshold_factors_file, max_threshold, factors)
    print(*factors)
    factor_architecture = FactorArchitecture(dim=num_factors, factors=factors)

    factor_architecture.get_factor_topology_elements()

    if fcn_num == 3:
        shift_data_file = "f03_o.txt"
    elif fcn_num == 5:
        shift_data_file = "f05_op.txt"
        matrix_data_file = "f05_m.txt"
    elif fcn_num == 11:
        shift_data_file = "f11_op.txt"
        matrix_data_file = "f11_m.txt"
    elif fcn_num == 17:
        shift_data_file = "f17_op.txt"
    else:
        shift_data_file = "f20_o.txt"

    function = Function(function_number=fcn_num, lbound=lb, ubound=ub, shift_data_file=shift_data_file,
                        matrix_data_file=matrix_data_file)
    print(function)

    # # Instantiate and run the FEA
    fea = FEA(
        function=function,
        fea_runs=fea_runs,
        generations=generations,
        pop_size=pop_size,
        factor_architecture=factor_architecture,
        base_algorithm=PSO
    )
    fea.run(base_performance_result_dir, performance_result_file)
###############################################################################


# file_path = '/Users/xuyingwangswift/Desktop/FEA_PCA_AUTOENCODER/src/Data/Generated_data_dim10_row10000/ackley.csv'  # Replace with your actual file path
# X_std = load_and_prepare_pca_data(file_path)
# vectors =  get_eigenvectors(X_std)

# threshold, adjacency_list = find_largest_threshold_for_connected_graph(vectors)
# print(threshold)
# print(adjacency_list)
