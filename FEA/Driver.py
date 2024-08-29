import numpy as np
from pso import PSO
from FEA import FEA
from Function import Function
from FactorArchitecture import FactorArchitecture
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import networkx as nx
import csv
import os

from sklearn.impute import SimpleImputer


def load_and_prepare_pca_data(file_path):
    """
    Load and preprocess data.
    """
    df = pd.read_csv(file_path)

    # Scale the data
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df)
    return X_std


def plot_graph(G, filename, graph_path):
    """
    Plots the graph G, saves it to a specified path, and displays the plot.

    Parameters:
    G (networkx.Graph): The graph to be plotted.
    filename (str): The name for the saved plot file.
    """
    # Set the size of the plot
    plt.figure(figsize=(8, 6))
    # Draw the graph
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500,
            font_size=10, font_weight='bold')

    # Set the title of the plot
    plt.title(filename + " Eigenvector Overlap Graph")

    # Define the path where the graph will be saved
    save_path = os.path.join(graph_path + filename + ".png")

    # Save the plot to the specified path
    plt.savefig(save_path)

    # Display the plot
    plt.show()
    #plt.close()


def is_overlapping(eigenvectors, features_above_threshold, filename, graph_path):
    # Create a fully connected graph
    G = nx.Graph()

    # Add nodes for each eigenvector
    for i in range(len(eigenvectors)):
        G.add_node(i + 1)

    # Add edges based on shared features above the threshold
    for i in range(len(features_above_threshold)):
        for j in range(i + 1, len(features_above_threshold)):
            if any(feature in features_above_threshold[j] for feature in features_above_threshold[i]):
                G.add_edge(i + 1, j + 1)

    # Check if the graph is fully connected
    is_fully_connected = nx.is_connected(G)
    plot_graph(G, filename, graph_path)
    print("The graph is fully connected:", is_fully_connected)
    return is_fully_connected


def get_pca_factors(X_std, num_factors):
    """
    Perform PCA and extract factors.
    """
    pca_full = PCA()
    pca_full.fit(X_std)

    # Extracting the eigenvectors
    eigenvectors = pca_full.components_

    # # Your existing plotting code
    # plt.figure(figsize=(12, 6))
    # sns.heatmap(eigenvectors, cmap='viridis', annot=True)
    # plt.title('Feature Contributions to Principal Components')
    # plt.xlabel('Features')
    # plt.ylabel('Principal Components')

    # # Saving the figure
    #plt.savefig('/path/to/save/figure.png')  # Specify your desired path and file name
    # Display the plot
    plt.close()
    return eigenvectors


def get_selected_features(eigenvectors, threshold, filename, graph_path):
    # print("Original Eigenvectors:", eigenvectors[:1])
    # print("Original Eigenvectors len:", len(eigenvectors))

    # Remove the last feature from each eigenvector
    eigenvectors_without_last_feature = [vector[:-1] for vector in eigenvectors]
    # print(len(eigenvectors_without_last_feature))
    # print("Eigenvectors without last feature:", eigenvectors_without_last_feature[:1])

    # Your existing plotting code
    plt.figure(figsize=(12, 6))
    sns.heatmap(eigenvectors_without_last_feature, cmap='viridis', annot=True)
    plt.title('Feature Contributions to Principal Components')
    plt.xlabel('Features')
    plt.ylabel('Principal Components')

    # Saving the figure
    plt.savefig('figure.png')  # Specify your desired path and file name

    plt.show()
    #plt.close()

    # Get indices of features in each eigenvector that are above the threshold
    indices_of_features_above_threshold = [
        [index for index, feature in enumerate(vector) if abs(feature) >= threshold]
        for vector in eigenvectors_without_last_feature
    ]

    # Print the indices of features above the threshold for each eigenvector
    for eigenvector_index, feature_indices in enumerate(indices_of_features_above_threshold):
        print(f"Eigenvector {eigenvector_index + 1}: Indices of features above threshold: {feature_indices}")

    overlapping = is_overlapping(eigenvectors, indices_of_features_above_threshold, filename, graph_path)
    if overlapping:
        return indices_of_features_above_threshold
    else:
        return None


def remove_duplicate_sublists(list_of_lists):
    seen = set()
    unique_lists = []
    for sublist in list_of_lists:
        # Convert sublist to tuple for hashability
        tuple_sublist = tuple(sublist)
        # Add to unique_lists only if not seen before
        if tuple_sublist not in seen:
            seen.add(tuple_sublist)
            unique_lists.append(sublist)
    return unique_lists


def write_global_fitness_to_csv(global_fitness_list, target_directory, file_name):
    """
    Write global fitness values to a CSV file.
    """
    file_path = os.path.join(target_directory, file_name)
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        rows = [[fitness] for fitness in global_fitness_list]
        writer.writerows(rows)


def get_fea_factor_architecture(data_file_path, num_factors, threshold, function_name, graph_path):
    # Load and preprocess data
    X_std = load_and_prepare_pca_data(data_file_path)

    # Perform PCA and extract factors
    factors = get_pca_factors(X_std, num_factors)
    # Printing the factors in a formatted way
    # print("PCA Factors:")
    # print(factors)

    selected_factors = get_selected_features(factors, threshold, function_name, graph_path)
    # print(selected_factors)

    no_duplicates_factors = remove_duplicate_sublists(selected_factors)
    # print(no_duplicates_factors)

    # Define the factor architecture
    factor_architecture = FactorArchitecture(dim=num_factors, factors=no_duplicates_factors)

    # print(factor_architecture)
    return factor_architecture


def get_function(fcn_num, lb, ub, fcn_name):
    # Define the objective function
    function = Function(function_number=fcn_num, lbound=lb, ubound=ub,
                        shift_data_file="f11_op.txt", matrix_data_file="f11_m.txt")
    print(function)
    return function


def run_fea_process(data_file_path, num_factors,
                    fea_runs, generations,
                    pop_size, function_name,
                    fcn_num, lb, ub,
                    performance_result_dir,
                    performance_result_file,
                    graph_path,
                    threshold):
    """
    Function to setup and run the Factored Evolutionary Algorithm process.
    """
    # Define the factor architecture
    factor_architecture = get_fea_factor_architecture(data_file_path, num_factors, threshold, function_name, graph_path)

    function = get_function(fcn_num, lb, ub, function_name)

    # # Instantiate and run the FEA
    fea = FEA(
        function=function,
        fea_runs=fea_runs,
        generations=generations,
        pop_size=pop_size,
        factor_architecture=factor_architecture,
        base_algorithm=PSO
    )
    fea.run(performance_result_dir, performance_result_file)

    def classic_random_grouping(self, group_size, overlap=True):
        """
        Random grouping as defined by Yang et al.
        Uses pre-defined group size to create distinct groupings, where variables are randomly added to groups.

        """
        if overlap:
            self.method = "classic_random_overlap_" + str(group_size) + "_" + str(group_size)
        else:
            self.method = "classic_random_" + str(group_size)
        number_of_groups = int(self.dim / group_size)
        indeces = list(range(0, self.dim))
        factors = []
        for n in range(number_of_groups - 1):
            grp = random.sample(indeces, k=group_size)
            factors.append(grp)
            for grpidx in grp:
                indeces.remove(grpidx)
        factors.append(indeces)
        if overlap:
            disjoint_length = len(factors)  # number of factors after disjoint grouping
            halfsize = int(group_size / 2)  # how many variables need to be selected from each factor to create overlap
            for i in range(disjoint_length):
                if i < disjoint_length - 1:  # stop before getting to last disjoint group, since this will be included already
                    new_factor = random.sample(factors[i], k=halfsize)
                    new_factor.extend(random.sample(factors[i + 1],
                                                    k=group_size - halfsize))  # if group size is odd, make sure overlapping groups have the same size
                    factors.append(new_factor)
        self.factors = factors


def main():
    # List of benchmark functions and their domain ranges
    # benchmark_functions = [
    #     ('ackley', (-32, 32)),
    #     ('dixon_price', (-10, 10)),
    #     ('exponential', (-1, 1)),
    #     ('griewank', (-100, 100)),
    #     ('powell_singular', (-4, 5)),
    #     ('rana', (-500, 500)),
    #     ('rastrigin', (-5.12, 5.12)),
    #     ('rosenbrock', (-2.048, 2.048)),
    #     ('schwefel', (-512, 512)),
    #     ('sphere', (-5.12, 5.12))
    # ]

    # Base paths
    base_data_path = "/Users/ashfak/workspace/FEA_PCA_AUTOENCODER/Data/Generated_data_dim30_row25000/"
    base_performance_result_dir = "/Users/ashfak/Desktop/FEA_PCA_AUTOENCODER/src/Results/dim100_gen25000_pca/"
    base_graph_path = "/Users/ashfak/workspace/FEA_PCA_AUTOENCODER/Graphs/"

    # Common parameters
    num_factors = 50
    fea_runs = 50
    generations = 100
    pop_size = 100
    # threshold = 0.10
    #
    # function_name = 'D_2m_group_shifted_m_rotated_ackley'
    # fcn_num = 11
    # lb = -32
    # ub = 32
    # # Define file paths
    # data_file_path = base_data_path + function_name + ".csv"
    # performance_result_file = function_name + '_' + threshold + '_data_dim_30_gen_25000_result.csv'
    #
    # # ANSI escape codes for color (e.g., green)
    # GREEN = "\033[92m"
    # RESET = "\033[0m"
    #
    # # Your print statement with color
    # print(f"{GREEN}Running FEA process for {function_name} (Function #{fcn_num}){RESET}")
    # # Call the FEA process function
    # run_fea_process(data_file_path, num_factors,
    #                 fea_runs, generations,
    #                 pop_size, function_name,
    #                 fcn_num, lb, ub,
    #                 base_performance_result_dir,
    #                 performance_result_file,
    #                 base_graph_path,
    #                 threshold)

    thresholds = [0.33]

    for threshold in thresholds:
        function_name = 'D_2m_group_shifted_m_rotated_ackley'
        fcn_num = 11
        lb = -32
        ub = 32
        # Define file paths
        data_file_path = base_data_path + function_name + ".csv"
        performance_result_file = f"{function_name}_{threshold}_data_dim_30_gen_25000_result.csv"

        # ANSI escape codes for color (e.g., green)
        GREEN = "\033[92m"
        RESET = "\033[0m"

        # Your print statement with color
        print(f"{GREEN}Running FEA process for {function_name} (Function #{fcn_num}) with threshold {threshold}{RESET}")

        # Call the FEA process function
        run_fea_process(data_file_path, num_factors,
                        fea_runs, generations,
                        pop_size, function_name,
                        fcn_num, lb, ub,
                        base_performance_result_dir,
                        performance_result_file,
                        base_graph_path,
                        threshold)


if __name__ == "__main__":
    main()
