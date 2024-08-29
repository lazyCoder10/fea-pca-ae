from autoencoder_grouping import run_autoencoder_fea_process
    
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
        ('powell_sum',     (-1, 1)),
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
    print(PINK + f"EA 30 is processing...")
    # List of benchmark functions and their domain ranges
    # Base paths
    base_data_path = "/Users/xuyingwangswift/Desktop/AE_FEA/Generated_data_dim30_row25000/"
    base_performance_result_dir = "/Users/xuyingwangswift/Desktop/AE_FEA/Results/AUTO_dim30"
    factor_dir = '/Users/xuyingwangswift/Desktop/AE_FEA/Results/AUTO_Factor_D30/'

    # Common parameters
    num_factors = 30
    fea_runs = 50
    generations = 100
    pop_size = 100

    # Looping through the benchmark_functions to find the specific function and its bounds
    for i, (func, bounds) in enumerate(benchmark_functions):
        function_name = func        
        fcn_num = i + 1  # Assuming the function number is its position in the list
        lb, ub = bounds

        # Define file paths
        data_file_path = base_data_path + function_name + ".csv"
        performance_result_file = function_name + '_ea_data_dim_30_gen_25000_result.csv'
        autofacors = function_name + '_dim30_autofactor.csv'
    
        # Your print statement with color
        print(f"{GREEN}Running FEA process for {function_name} (Function #{fcn_num}){RESET}")
        # Call the FEA process function
        run_autoencoder_fea_process(data_file_path,
                                    num_factors, 
                                    fea_runs, 
                                    generations, 
                                    pop_size,
                                    fcn_num, lb, ub,
                                    base_performance_result_dir,
                                    performance_result_file,
                                    factor_dir,
                                    autofacors)
        print(PINK + f"EA 30 has completed.")
autoencoder()