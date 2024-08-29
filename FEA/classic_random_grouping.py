from random import random
from FEA import FEA
from FactorArchitecture import FactorArchitecture
import random
from Function import Function
from pso import PSO

def classic_random_grouping(N, M):
    """
    Create overlapping random groupings. Each group will have a pre-defined size and 
    variables can be randomly added to multiple groups.

    Args:
    N (int): Total number of variables.
    M (int): Size of each group.

    Returns:
    list of lists: A list of groups, each group is a list of variable names.
    """
    # Create a list of all variables
    variables = list(range(N))

    # Initialize list to store groups
    groups = []

    # Create groups of size M with possible overlaps
    for _ in range(N):
        group = random.sample(variables, M)
        groups.append(group)

    return groups


def run_random_fea_process(dim, 
                    fea_runs, generations, 
                    pop_size,
                    fcn_num, lb, ub,
                    performance_result_dir,
                    performance_result_file):
    """
    Function to setup and run the Factored Evolutionary Algorithm process.
    """ 

    #factors = classic_random_grouping(dim, 5)
    factors = classic_random_grouping(dim, dim//2)
    print(factors)

    # Define the factor architecture
    fa = FactorArchitecture(dim, factors)
    fa.get_factor_topology_elements()

    function = Function(function_number=fcn_num, lbound=lb, ubound=ub, shift_data_file="f17_op.txt",
                        matrix_data_file="")
    print(function)
    # # Instantiate and run the FEA
    fea = FEA(
        function=function,
        fea_runs=fea_runs,    
        generations=generations,
        pop_size=pop_size,
        factor_architecture=fa,
        base_algorithm=PSO
    )
    fea.run(performance_result_dir, performance_result_file)

