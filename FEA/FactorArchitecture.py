class FactorArchitecture(object):
    """
    Manages the topology for a Factored Evolutionary Algorithm (FEA). This topology defines 
    how variables (dimensions) and swarms (subpopulations) are organized and interact with each other.

    The architecture involves categorizing variables and swarms into groups and determining their 
    relationships. This structuring is essential for the algorithm's divide-and-conquer approach.
    """

    def __init__(self, dim=0, factors=None):
        """
        Initializes the FactorArchitecture object.

        :param dim: The number of dimensions (variables) of the problem. It sets the problem's scale.
        :param factors: Pre-defined factor groups. Each factor is a list of variables grouped together. 
                        If provided, these factors are used to initialize the factor architecture.
        """
        # Arbiters, optimizers, and neighbors are key components of the architecture
        self.arbiters = []    # Arbiters decide which swarm has control over a particular variable.
        self.optimizers = []  # Optimizers are swarms responsible for optimizing specific variables.
        self.neighbors = []   # Neighbors represent the swarms that share variables and thus need to coordinate.
        
        self.dim = dim        # The problem's dimensionality.
        self.method = ""      # Stores the method used to generate the topology.
        self.function_evaluations = 0  # Counter for the number of function evaluations.

        # If pre-defined factors are provided, use them to set up the architecture.
        if factors is not None:
            self.factors = factors
            self.get_factor_topology_elements()
        else:
            # If no factors are provided, initialize to an empty list.
            self.factors = []
    
    def __str__(self):
        """
        Provides a string representation of the FactorArchitecture object.

        :return: A string detailing the attributes of the FactorArchitecture.
        """
        description = f"Factor Architecture:\n"
        description += f"  Dimensions: {self.dim}\n"
        description += f"  Method: {self.method}\n"
        description += f"  Function Evaluations: {self.function_evaluations}\n"
        description += f"  Factors: {self.factors}\n"
        description += f"  Arbiters: {self.arbiters}\n"
        description += f"  Optimizers: {self.optimizers}\n"
        description += f"  Neighbors: {self.neighbors}\n"
        return description

    def get_factor_topology_elements(self):
        """
        Calculates and assigns the arbiters, optimizers, and neighbors for the factor architecture.

        This method is crucial in defining the relationships and interactions between the different subpopulations 
        (referred to as swarms) and the variables they are optimizing. It ensures that the factor architecture 
        is fully defined with all necessary components for the FEA algorithm to function effectively.
        """

        # Nominate arbiters for each variable.
        # Arbiters are responsible for deciding which swarm (subpopulation) has control over each variable.
        # This step assigns a specific swarm as the arbiter for each variable based on the factor architecture.
        self.nominate_arbiters()

        # Calculate optimizers for each variable.
        # Optimizers are swarms that are assigned the task of optimizing specific variables.
        # This step identifies which swarms are responsible for optimizing each variable.
        self.calculate_optimizers()

        # Determine neighbors for each swarm.
        # Neighbors are swarms that share variables and hence have potential interactions or dependencies.
        # This step identifies the neighboring swarms for each swarm, facilitating coordination in optimization tasks.
        self.determine_neighbors()

    def nominate_arbiters(self):
        """
        Assigns arbiters for each variable in the factor architecture. An arbiter is a specific swarm (subpopulation)
        responsible for a particular variable. This method ensures that each variable in the problem has a designated
        swarm that will make decisions about its optimization.

        The method uses the structure of the factors to determine which swarm becomes the arbiter for each variable.
        """

        assignments = {}  # A dictionary to hold the assignments of arbiters.

        # Create a shallow copy of the factors list for faster iteration.
        factors = [f for f in self.factors]

        # Iterate over all factors except the last one.
        for i, factor in enumerate(factors[:-1]):
            for j in factor:
                # Assign the current swarm as the arbiter for a variable if:
                # 1. The variable is not in the next factor (ensuring uniqueness of assignment).
                # 2. The variable has not already been assigned an arbiter.
                if j not in self.factors[i + 1] and j not in assignments:
                    assignments[j] = i

        # Handle the last factor separately as it doesn't have a subsequent factor to compare with.
        for j in factors[-1]:
            # Assign the last swarm as the arbiter for any remaining variables.
            if j not in assignments:
                assignments[j] = len(factors) - 1

        # Sort the keys (variable indices) to maintain a consistent order.
        keys = list(assignments.keys())
        keys.sort()

        # Create the final list of arbiters based on the sorted keys.
        arbiters = [assignments[k] for k in keys]

        # Assign the computed arbiters to the class attribute.
        self.arbiters = arbiters

    def calculate_optimizers(self):
        """
        Identifies and assigns optimizers for each variable in the factor architecture. 
        An optimizer in this context refers to a list of swarms (subpopulations) responsible 
        for optimizing a specific variable. This method is crucial for the Factored Evolutionary 
        Algorithm (FEA) to ensure that each variable in the problem domain is assigned to one or 
        more swarms for optimization.

        :return: None. The method updates the class attribute 'optimizers'.
        """

        # Initialize an empty list to store the optimizers for each variable.
        optimizers = []

        # Create a shallow copy of the factors list for more efficient iteration.
        factors = [f for f in self.factors]

        # Iterate over all variables in the problem domain.
        for v in range(self.dim):
            # Initialize an empty list to store the indices of swarms optimizing the current variable.
            optimizer = []

            # Iterate over each factor (group of variables optimized by a specific swarm).
            for i, factor in enumerate(factors):
                # Check if the current variable is part of the current factor.
                if v in factor:
                    # If the variable is part of the factor, add the index of the swarm (factor's index) to the optimizer list.
                    optimizer.append(i)

            # Append the list of swarms (optimizer) for the current variable to the main optimizers list.
            optimizers.append(optimizer)

        # Update the class attribute 'optimizers' with the list of optimizers for each variable.
        self.optimizers = optimizers
       
    def determine_neighbors(self):
        """
        Identifies and assigns neighbors for each swarm in the factor architecture. 
        Neighbors are defined as swarms that share at least one variable in common, 
        indicating a potential interaction or dependency between them. This method 
        helps to map out these relationships, which are important for coordinating 
        optimization efforts in a Factored Evolutionary Algorithm (FEA).

        :return: None. The method updates the class attribute 'neighbors'.
        """

        # Initialize an empty list to store the neighbors for each swarm.
        neighbors = []

        # Create a shallow copy of the factors list for more efficient iteration.
        factors = [f for f in self.factors]

        # Iterate over each factor (swarm) to determine its neighbors.
        for i, factor in enumerate(factors):
            # Initialize an empty list to store the indices of neighboring swarms for the current swarm.
            neighbor = []

            # Iterate over all other factors to check for shared variables.
            for j, other_factor in enumerate(factors):
                # Check if the current swarm (i) is different from the other swarm (j) and they share at least one variable.
                if (i != j) and not set(factor).isdisjoint(set(other_factor)):
                    # If they share at least one variable, add the index of the other swarm to the neighbor list.
                    neighbor.append(j)

            # Append the list of neighboring swarms for the current swarm to the main neighbors list.
            neighbors.append(neighbor)

        # Update the class attribute 'neighbors' with the list of neighbors for each swarm.
        self.neighbors = neighbors

