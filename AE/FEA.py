import numpy as np
import csv
import os

Magenta = "\033[35m"
Cyan = "\033[36m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
ENDC = "\033[0m" 

class FEA:
    """
    Factored Evolutionary Algorithm (FEA) for single-objective optimization.
    This class integrates a factor-based approach with an evolutionary algorithm
    to optimize a given objective function.
    """

    def __init__(self, function, fea_runs, generations, pop_size, factor_architecture, base_algorithm, continuous=True, seed=None):
        """
        Initializes the FEA.

        :param function: The objective function to be optimized.
        :param fea_runs: Number of FEA iterations to perform.
        :param generations: Number of generations for the base algorithm.
        :param pop_size: Population size for each subpopulation.
        :param factor_architecture: An instance of FactorArchitecture.
        :param base_algorithm: The base evolutionary algorithm to be used.
        :param continuous: A flag indicating whether the problem is continuous.
        :param seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        self.function = function
        self.fea_runs = fea_runs
        self.base_alg_iterations = generations
        self.pop_size = pop_size
        self.factor_architecture = factor_architecture
        self.dim = factor_architecture.dim
        self.base_algorithm = base_algorithm
        self.global_solution = None
        self.global_fitness = np.inf
        self.solution_history = []
        self.set_global_solution(continuous)
        self.subpopulations = self.initialize_factored_subpopulations()
        self.global_fitness_list = []

    def run(self, result_dir, filename):
        """
        Executes the FEA process for a specified number of runs.

        In each FEA run, the method iterates through all the subpopulations,
        allowing each to undergo an optimization process using the base algorithm.
        After all subpopulations are processed, the compete and share_solution methods
        are called to integrate the results across the subpopulations. The global
        fitness value is also printed to track the progress of the algorithm.

        The compete method allows variables to compete across factors to find the 
        best values for each variable, thereby creating an optimized global solution.
        The share_solution method then shares this global solution back with the 
        subpopulations for their next run, ensuring that the entire population moves 
        towards better solutions over time.
        """
        
        # Ensure the directory exists
        os.makedirs(result_dir, exist_ok=True)

        # Construct the full path for the file
        file_path = os.path.join(result_dir, filename)
        
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['FEA Run', 'Global Fitness'])  # Writing the header
            
            for fea_run in range(self.fea_runs):

                print(GREEN + f"Starting FEA run {fea_run+1} of {self.fea_runs}" + ENDC)
                # Iterate through each subpopulation and run their respective optimization algorithm
                for alg in self.subpopulations:
                    alg.run()  # Run the base algorithm for the current subpopulation

                print(Magenta + f"Competing subpopulations for FEA run {fea_run+1}" + ENDC)
                # Compete step: optimizes each variable across all subpopulations
                self.compete()

                print(BLUE + f"Sharing solution among subpopulations for FEA run {fea_run+1}" + ENDC)
                # Share step: updates each subpopulation with the latest global solution
                self.share_solution()

                # Print the current FEA run number and the global fitness value
                print(GREEN + f'FEA run {fea_run+1} completed. Global Fitness: {self.global_fitness}')
                
                # Writing the FEA run number and the global fitness value to the CSV file
                writer.writerow([fea_run + 1, self.global_fitness])

                self.global_fitness_list.append(self.global_fitness)


    def set_global_solution(self, continuous):
        """
        Initializes the global solution based on the problem type (continuous or discrete).

        This method sets up the initial global solution, which is a single vector
        representing a candidate solution for the optimization problem. The method also
        evaluates the fitness of this initial solution and records it in the global fitness
        attribute. Additionally, the initial solution is added to the solution history
        for tracking the evolution of solutions over time.

        :param continuous: A boolean flag indicating whether the problem is continuous.
                        If True, the global solution will be initialized with random
                        values within the defined bounds of the problem.
        """
        if continuous:
            # If the problem is continuous, initialize the global solution with random values.
            # The random values are uniformly distributed within the lower and upper bounds
            # of the problem's search space (defined by self.function.lbound and ubound).
            self.global_solution = np.random.uniform(self.function.lbound, self.function.ubound, size=self.dim)

            # Evaluate the fitness of the initial global solution using the objective function.
            # This fitness value is used to assess the quality of the solution.
            self.global_fitness = self.function.run(self.global_solution)

            # Append the initial solution to the solution history.
            # The solution history keeps track of all solutions generated during the FEA process.
            self.solution_history.append(self.global_solution)


    def initialize_factored_subpopulations(self):
        """
        Initializes the subpopulations for the FEA algorithm based on the provided factor architecture.

        This method creates a subpopulation for each factor defined in the factor architecture. 
        Each subpopulation is tasked with optimizing a specific subset of the problem variables, 
        as indicated by its corresponding factor. The method initializes each subpopulation 
        using the base evolutionary algorithm provided.

        The initialization includes setting the objective function, the dimensions (number of variables) 
        for each subpopulation, the number of generations, the population size, the specific factor 
        (subset of variables), and the initial global solution.

        :return: A list of initialized subpopulation algorithms, one for each factor.
        """
        # Iterate through each factor in the factor architecture
        return [
            # Initialize a subpopulation algorithm for each factor
            self.base_algorithm(
                function=self.function,             # The objective function for optimization
                dim=len(factor),                    # The number of variables in this factor
                generations=self.base_alg_iterations, # The number of generations for the base algorithm
                population_size=self.pop_size,      # The population size for each subpopulation
                factor=factor,                      # The specific subset of variables (factor) to optimize
                global_solution=self.global_solution # The initial global solution to use in the subpopulation
            )
            for factor in self.factor_architecture.factors # Iterate through all factors
        ]


    def share_solution(self):
        """
        Shares the current global solution with each subpopulation in the FEA algorithm.

        This method is responsible for disseminating the information from the global solution 
        to each subpopulation. It updates the individuals within each subpopulation based on 
        the global solution, allowing them to integrate the latest global insights into their 
        local search processes. This step is crucial for maintaining coherence and coordination 
        among the subpopulations, ensuring they collectively work towards improving the global solution.

        Each subpopulation's individuals are updated to reflect the global solution, and the 
        worst solutions in each subpopulation are replaced with this global solution. This 
        replacement helps in steering the entire population towards promising areas in the 
        solution space.
        """
        for alg in self.subpopulations:
            # Update each individual in the subpopulation based on the global solution.
            # This typically involves evaluating the fitness of the individual with respect to the global solution
            # and potentially updating certain attributes or behaviors based on this global context.
            alg.pop = [individual.update_individual_after_compete(self.global_solution) for individual in alg.pop]

            # Replace the worst solution in the subpopulation with the current global solution.
            # This step helps to ensure that even the least performing parts of the population are aligned
            # with the overall direction of the global optimization process.
            alg.replace_worst_solution(self.global_solution)


    def compete(self):
        """
        Executes the competition step of the FEA algorithm.

        During the compete step, each variable in the global solution is optimized across all relevant subpopulations.
        This process involves finding the best value for each variable from the subpopulations where it is included and
        updating the global solution with these optimized values. This method ensures that the best findings from
        localized searches are integrated into the overall solution.

        After updating the global solution with the best values for each variable, the global fitness is recalculated,
        and the updated global solution is stored in the solution history for tracking the progress over time.
        """
        # Iterate through each variable in the global solution
        for var_idx in range(self.dim):
            # Evaluate and find the best value for the current variable across all subpopulations
            best_value_for_var = self.evaluate_best_value_for_variable(var_idx)

            # Update the global solution with the best value found for this variable
            self.global_solution[var_idx] = best_value_for_var

        # Recalculate the global fitness of the updated global solution
        self.global_fitness = self.function.run(self.global_solution)

        # Append the updated global solution to the solution history for tracking
        self.solution_history.append(self.global_solution)


    def evaluate_best_value_for_variable(self, var_idx):
        """
        Evaluates and identifies the best value for a specific variable across all relevant subpopulations.

        This method is a key component of the compete step in FEA. It assesses each variable's
        performance (based on the objective function) across different subpopulations and identifies
        the value that yields the best fitness. The aim is to find the most optimal value for a variable
        considering its impact on the global solution.

        :param var_idx: Index of the variable in the global solution to be evaluated.
        :return: The best value found for the variable across all subpopulations.
        """
        # Initialize best value as the current value of the variable in the global solution
        best_value = self.global_solution[var_idx]

        # Record the current fitness of the global solution
        current_fitness = self.function.run(self.global_solution)

        # Iterate over subpopulations that include the variable
        for pop_idx in self.factor_architecture.optimizers[var_idx]:
            curr_pop = self.subpopulations[pop_idx]
            
            # Find the index of the variable in the subpopulation
            pop_var_idx = np.where(curr_pop.factor == var_idx)[0][0]

            # Retrieve the candidate value for the variable from the subpopulation's best solution
            candidate_value = curr_pop.gbest.position[pop_var_idx]

            # Create a new solution by replacing the variable's value in the global solution
            new_solution = self.global_solution.copy()
            new_solution[var_idx] = candidate_value

            # Evaluate the fitness of the new solution
            new_fitness = self.function.run(new_solution)

            # If the new solution has better fitness, update the best value and current fitness
            if new_fitness < current_fitness:
                best_value = candidate_value
                current_fitness = new_fitness

        # Return the best value found for the variable
        return best_value


