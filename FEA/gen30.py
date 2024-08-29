from classic_random_grouping import run_random_fea_process
from pca_grouping import run_pca_fea_process
import math
   
# ANSI escape codes for color (e.g., green)
GREEN = "\033[92m"
RESET = "\033[0m"
YELLOW = "\033[93m"
RED = "\033[91m"
PINK = "\033[95]"

benchmark_functions = [
        ('ackley', (-32, 32)),
        ('brown', (-1, 4))
        # ('dixon_price', (-10, 10)),
        # ('griewank', (-100, 100)),
        # ('powell_singular', (-4, 5)),
        # ('powell_singular2', (-4, 5)),
        # ('powell_sum',     (-1, 1)),
        # ('qing_function', (-500, 500)),
        # ('quartic_function', (-1.28, 1.28)),
        # ('rastrigin', (-5.12, 5.12)),
        # ('rosenbrock', (-2.048, 2.048)),
        # ('salomon', (-100, 100)),
        # ('sargan_function', (-100,100)),
        # ("schumer_steiglitz", (-100, 100)),
        # ("schwefel_1_2", (-100, 100)),
        # ("sphere", (-5.12, 5.12)),
        # ("step_function", (-100,100)),
        # ("streched_v_sine_wave", (-10, 10)),
        # ("trigonometric_1", (0, math.pi)),
        # ("trigonometric_2", (-500, 500)),
        # ("weierstrass", (-0.5, 0.5))
    ]

def check_function(function_name, fcn_num, lb, ub, benchmark_functions):
    for index, (func, bounds) in enumerate(benchmark_functions):
        if func == function_name and index + 1== fcn_num:
            if bounds[0] == lb and bounds[1] == ub:
                return True
    return False

def pca():
    print(YELLOW + f"PCA 30 is processing... ")
 
    # Base paths
    base_data_path = "/Users/xuyingwangswift/Desktop/FEA_PCA_AUTOENCODER/src/Data/Generated_data_dim30_row25000/"
    base_performance_result_dir = "/Users/xuyingwangswift/Desktop/FEA_PCA_AUTOENCODER/src/Results/PCA_dim30"
    threshold_dir = '/Users/xuyingwangswift/Desktop/FEA_PCA_AUTOENCODER/src/Results/Threshold/'

    # Common parameters
    num_factors = 30
    fea_runs = 50
    generations = 100
    pop_size = 100

    function_name = 'brown'
    fcn_num = 2
    lb = -1
    ub = 4
    # Define file paths
    data_file_path = base_data_path + function_name + ".csv"
    performance_result_file = function_name + '_pca_data_dim_30_gen_25000_result.csv'
    max_threshold_factors_file = function_name + '_dim30_threshold.csv'
    
    # Your print statement with color
    print(f"{GREEN}Running FEA process for {function_name} (Function #{fcn_num}){RESET}")

    match = check_function(function_name, fcn_num, lb, ub, benchmark_functions)

    if (match):
        # Call the FEA process function
        run_pca_fea_process(data_file_path, num_factors, 
                            fea_runs, generations, 
                            pop_size,
                            fcn_num, lb, ub,
                            base_performance_result_dir,
                            performance_result_file,
                            threshold_dir,
                            max_threshold_factors_file)
    else:
        print(f"{RED}Parameters for {function_name} (Function #{fcn_num}){RED} do not match")

    print(f"PCA 30 has completed." + RESET)


def random_fea():
    print(YELLOW + f"Random 30 is processing...")

    # Base paths
    base_performance_result_dir = "/Users/xuyingwangswift/Desktop/FEA_PCA_AUTOENCODER/src/Results/Random_dim30"

    # Common parameters
    dim = 30
    fea_runs = 50
    generations = 100
    pop_size = 100
    
    function_name = 'ackley'
    fcn_num = 1
    lb = -32
    ub = 32
    # Define file paths
    performance_result_file = function_name + '_random_data_dim_30_gen_25000_result.csv'

    # Your print statement with color
    print(f"{GREEN}Running FEA process for {function_name} (Function #{fcn_num}){RESET}")
        # Call the FEA process function

    match = check_function(function_name, fcn_num, lb, ub, benchmark_functions)
    if (match):
        run_random_fea_process(dim, 
                            fea_runs, generations, 
                            pop_size,
                            fcn_num, lb, ub,
                            base_performance_result_dir,
                            performance_result_file)
    print(f"Random 30 has completed."+ RESET)

def main():
    #print ("START PCA")
    #pca()
    print(PINK + f"START RANDOM")
    random_fea()
main()