from classic_random_grouping import run_random_fea_process

# ANSI escape codes for color (e.g., green)
GREEN = "\033[92m"
RESET = "\033[0m"
PINK = "\033[95]"
YELLOW = "\033[93m"
RED = "\033[91m"

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
    ("weierstrass", (-0.5, 0.5)),
    ("zakharov", (-5, 10))
]


def random_fea():
    print(PINK + f"Radom 50 processing ....")

    # Base paths
    base_data_path = "/Users/ashfak/workspace/FEA_PCA_AUTOENCODER/Data/Generated_data_dim50_row50000/"

    threshold_dir = '/Users/ashfak/Desktop/FEA_PCA_AUTOENCODER/src/Results/Threshold/'

    # Common parameters
    dim = 250
    fea_runs = 50
    generations = 100
    pop_size = 100

    # Looping through the benchmark_functions to find the specific function and its bounds
    # for i, (func, bounds) in enumerate(benchmark_functions):
    function_name = "shifted_rosenbrock"
    fcn_num = 20  # Assuming the function number is its position in the list
    lb = -100
    ub = 100
    # Define file paths
    performance_result_file = function_name + f'_random_data_dim{dim}_gen_{dim * 1000}_result.csv'
    base_performance_result_dir = f"/home/m33w398/FEA_PCA_AE/Random_Results/"
    # Your print statement with color
    print(f"{GREEN}Running FEA process for {function_name} (Function #{fcn_num}){RESET}")
    run_random_fea_process(dim,
                           fea_runs, generations,
                           pop_size,
                           fcn_num, lb, ub,
                           base_performance_result_dir,
                           performance_result_file)
    print(PINK + f"Random 50 Completed.")


random_fea()
