from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from helper import get_scaled_data
from FEA import FEA
from FactorArchitecture import FactorArchitecture
from Function import Function
from pso import PSO
import numpy as np
import networkx as nx
import os
import csv


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

    sorted_values = np.unique(np.sort((all_values)))
    print(sorted_values)

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


#############################################################################

def autoencoder_grouping(file_path, encoding_dim, activation, reg):
    scaled_data, y = get_scaled_data(file_path)
    # If your goal is feature learning or data denoising, 
    # where maintaining the data structure is important, you might choose an encoding_dim close to 10. 
    # This would not compress your data much but can help the model learn a useful representation of your data.#

    # this is the size of our encoded representations

    input_dim = scaled_data.shape[1]
    # this is our input placeholder
    input_img = Input(shape=(input_dim,))

    """
    Feature Extraction: The encoded representations (the output of the encoder model) 
    can be used as new features for downstream tasks, such as classification or regression.
    """
    # "encoded" is the encoded representation of the input, with L1 regularization
    if reg == 'l1':
        encoded = Dense(encoding_dim, activation=activation, activity_regularizer=regularizers.l1(1e-6))(input_img)
    else:
        encoded = Dense(encoding_dim, activation=activation, activity_regularizer=regularizers.l2(1e-4))(input_img)

    # Add the decoder part to complete the autoencoder
    decoded = Dense(input_dim, activation=activation)(encoded)
    autoencoder = Model(input_img, decoded)
    # Compile and train the autoencoder
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=256, shuffle=True)
    # Create a model that takes the input and outputs the encoded features
    encoder_model = Model(inputs=input_img, outputs=encoded)

    # Check if the Dense layer has weights
    if len(encoder_model.layers) > 0:
        dense_layer = encoder_model.layers[1]  # Accessing the Dense layer
        try:
            dense_layer_weights = dense_layer.get_weights()[0]
            print("Weights of the Dense layer:", dense_layer_weights)
        except IndexError:
            print("An error occurred while accessing the weights of the Dense layer.")
    else:
        print("No layers with weights found in the model.")
    abs_weights = np.abs(dense_layer_weights)
    print(abs_weights)
    t, factors = find_largest_threshold_for_connected_graph(abs_weights)

    return t, factors


def run_autoencoder_fea_process(data_file_path,
                                num_factors,
                                fea_runs,
                                generations,
                                pop_size,
                                fcn_num, lb, ub,
                                base_performance_result_dir,
                                performance_result_file,
                                factor_dir,
                                autofacors, activation, reg):
    # factors = classic_random_grouping(dim, 5)

    shift_data_file = ""
    matrix_data_file = ""

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

    t, factors = autoencoder_grouping(data_file_path, num_factors, activation, reg)

    write_to_file(factor_dir, autofacors, t, factors)
    # Define the factor architecture
    fa = FactorArchitecture(num_factors, factors)
    fa.get_factor_topology_elements()

    function = Function(function_number=fcn_num, lbound=lb, ubound=ub,
                        shift_data_file=shift_data_file,
                        matrix_data_file=matrix_data_file)
    print(function)
    # Instantiate and run the FEA
    fea = FEA(
        function=function,
        fea_runs=fea_runs,
        generations=generations,
        pop_size=pop_size,
        factor_architecture=fa,
        base_algorithm=PSO
    )
    fea.run(base_performance_result_dir, performance_result_file)
