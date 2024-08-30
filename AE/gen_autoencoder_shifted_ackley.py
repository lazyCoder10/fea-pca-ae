from autoencoder_grouping import run_autoencoder_fea_process
from huggingface_hub import hf_hub_download
import pandas as pd

# ANSI escape codes for color (e.g., green)
GREEN = "\033[92m"
RESET = "\033[0m"
PINK = "\033[95]"

benchmark_functions = [
    ('ackley', (-32, 32)),
    ('brown', (-1, 4)),
    ('dixon_price', (-10, 10)),
    ('griewank', (-100, 100)),
    ('powell_singular', (-4, 5)),
    ('powell_singular2', (-4, 5)),
    ('powell_sum', (-1, 1)),
    ('qing_function', (-500, 500)),
    ('quartic_function', (-1.28, 1.28)),
    ('rastrigin', (-5.12, 5.12)),
    ('rosenbrock', (-2.048, 2.048)),
    ('salomon', (-100, 100)),
    ("schwefel", (-100, 100)),
    ("schwefel_1_2", (-100, 100)),
    ("schwefel_2_20", (-100, 100)),
    ("sphere", (-5.12, 5.12)),
    ("stepint", (-5.12, 5.12)),
    ("sum_squares", (-10, 10)),
    ("zakharov", (-5, 10))
]


def autoencoder():
    print(PINK + f"AUTOENCODER 50 is processing...")
    # List of benchmark functions and their domain ranges
    # Base paths
    # base_data_path = "/Users/ashfak/workspace/FEA_PCA_AUTOENCODER/Data/Generated_data_dim100_row100000/"

    # Common parameters
    num_factors = 250
    fea_runs = 50
    generations = 100
    pop_size = 100

    # Looping through the benchmark_functions to find the specific function and its bounds
    # for i, (func, bounds) in enumerate(benchmark_functions):
    function_name = "shifted_ackley"
    fcn_num = 3  # Assuming the function number is its position in the list
    lb = -32
    ub = 32

    # Define file paths
    # data_file_path = base_data_path + function_name + ".csv"

    activation_func = ['linear', 'tanh', 'relu']
    regularizer = ['l1', 'l2']
    # activation_func = ['tanh', 'relu']
    # regularizer = ['l1', 'l2']

    base_performance_result_dir = f"/home/m33w398/FEA_PCA_AE/Results/dim_{num_factors}_{function_name}"
    factor_dir = f'/home/m33w398/FEA_PCA_AE/Factors/{function_name}'

    REPO_ID = "ashfakurarju/FEA-PCA-AE"
    FILENAME = "shifted_ackley.csv"

    dataset = pd.read_csv(
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    )

    # print(dataset)

    for activation in activation_func:
        for reg in regularizer:
            performance_result_file = f"{function_name}_{activation}_ae_{reg}_data_dim_{num_factors}_gen_{num_factors * 1000}_result.csv"
            autofacors = f"{function_name}_dim{num_factors}_{activation}_{reg}_factor.csv"

            # performance_result_file = function_name + '_linear_ae_l1_data_dim_100_gen_100000_result.csv'
            # autofacors = function_name + '_dim100_linear_l1_factor.csv'

            # Your print statement with color
            print(f"{GREEN}Running FEA process for {function_name} (Function #{fcn_num}){RESET}")
            # Call the FEA process function
            run_autoencoder_fea_process(dataset,
                                        num_factors,
                                        fea_runs,
                                        generations,
                                        pop_size,
                                        fcn_num, lb, ub,
                                        base_performance_result_dir,
                                        performance_result_file,
                                        factor_dir,
                                        autofacors, activation, reg)
            print(PINK + f"Autoencoder 50 has completed.")


autoencoder()
