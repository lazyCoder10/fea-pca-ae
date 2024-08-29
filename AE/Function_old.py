import numpy as np
import math
import random

class Function():
    def __init__(self, function_number, lbound, ubound):
        self.dimensions = 0
        self.lbound = lbound
        self.ubound = ubound
        self.function_to_call = 'F'+str(function_number)

        function_names = {
            1: "ackley",
            2: "brown",
            3: "dixon_price",
            4: "griewank",
            5: "powell_singular",
            6: "powell_singular2",
            7: "powell_sum",
            8: "qing_function",
            9: "quartic_function",
            10: "rastrigin",
            11: "rosenbrock",
            12: "salomon",
            13: "schwefel",
            14: "schwefel_1_2",
            15: "schwefel_2_20",
            16: "sphere",
            17: "stepint",
            18: "sum_squares",
            19: "zakharov"
        }

        self.name = function_names.get(function_number, "")
    
    def __str__(self):
        """
        Provides a string representation of the Function object.

        :return: A string detailing the attributes of the Function.
        """
        # ANSI escape codes for colors
        header_color = '\033[95m'  # Purple
        key_color = '\033[94m'  # Blue
        value_color = '\033[92m'  # Green
        end_color = '\033[0m'  # Reset color

        description = f"{header_color}Function Details:{end_color}\n"
        description += f"  {key_color}Name:{end_color} {value_color}{self.name}{end_color}\n"
        description += f"  {key_color}Lower Bound:{end_color} {value_color}{self.lbound}{end_color}\n"
        description += f"  {key_color}Upper Bound:{end_color} {value_color}{self.ubound}{end_color}\n"
        description += f"  {key_color}Function to Call:{end_color} {value_color}{self.function_to_call}{end_color}\n"
        return description

    def run(self, solution):
        """
        Executes the optimization function specified in 'self.function_to_call' on the provided solution.

        This method is the central point for computing the fitness of a solution based on the selected optimization function. 
        It dynamically calls the appropriate function within this class based on the 'self.function_to_call' attribute.

        :param solution: A numpy array or list representing a candidate solution to the optimization problem.
        :return: The computed fitness value of the provided solution.
        """

        # If the problem dimensions haven't been set yet (i.e., dimensions == 0),
        # determine and set the dimensions based on the length of the provided solution.
        if self.dimensions == 0:
            self.dimensions = len(solution)
            # Optionally, here you can include a check for the problem size to ensure it's within expected limits.

        # Dynamically call the function specified in 'self.function_to_call'.
        # 'getattr' is used to retrieve the function by its name (as a string) and then call it with the solution.
        # This approach provides flexibility, allowing the class to easily switch between different optimization functions.
        return getattr(self, self.function_to_call)(solution=solution)

        # function 1
    def F1(self, solution, name = "ackley"):
        if name == "ackley":
            # Define the constants a, b, and c for the Ackley function
            a = 20
            b = 0.2
            c = 2 * math.pi
            # Get the number of dimensions (n) from the length of the input array x
            n = len(solution)
            # Calculate the first part of the Ackley function
            sum1 = np.sum(solution ** 2)
            # Calculate the second part of the Ackley function
            sum2 = np.sum(np.cos(c * solution))
            # Calculate the individual terms for the Ackley function
            term1 = -a * math.exp(-b * math.sqrt(sum1 / n))
            term2 = -math.exp(sum2 / n)
            # Combine the terms and add constants to get the final result
            result = term1 + term2 + a + math.exp(1)
            return result
        else:
            # Handle other functions if needed
            raise ValueError("Unsupported function name")
    
    def F2(self, solution, name="brown"):
        """
        Computes the Brown function value for a given input vector.

        Args:
        solution (list or numpy array): Input vector.
        name (str): Name of the function, default is "Brown".

        Returns:
        float: The function value at the given solution.
        """
        if name.lower() != "brown":
            raise ValueError("This function is specifically implemented for the Brown function.")

        n = len(solution)
        sum = 0
        for i in range(n - 1):
            sum += (solution[i]**2)**(solution[i + 1]**2 + 1) + (solution[i + 1]**2)**(solution[i]**2 + 1)
        return sum

    def F3(self, solution, name="dixon_price"):
        if name == "dixon_price":
            n = len(solution) 
            # Check the dimension of the input vector
            if n < 1:
                raise ValueError("Dimension of the input vector must be at least 1.")
            result = (solution[0] - 1) ** 2
            for i in range(1, n):
                result += (i + 1) * (2 * solution[i] ** 2 - solution[i - 1]) ** 2
            return result
        else:
            # Handle other functions if needed
            raise ValueError("Unsupported function name")
    
    def F4(self, solution, name="griewank"):
        if name == "griewank":
            n = len(solution)
            sum_sq = np.sum(np.square(solution))
            prod_cos = np.prod(np.cos(solution / np.sqrt(np.arange(1, n + 1))))
            result = 1 + (sum_sq / 4000) - prod_cos
            return result
        else:
            # Handle other functions if needed
            raise ValueError("Unsupported function name")

    def F5(self, solution, name="powell_singular"):
        if name == "powell_singular":
            n = len(solution)
            if n < 4:
                raise ValueError("Dimension of the input vector must be at least 4.")
            result = 0
            for i in range(0, n - 3, 4):
                result += (solution[i] + 10 * solution[i + 1]) ** 2
                result += 5 * (solution[i + 2] - solution[i + 3]) ** 2
                result += (solution[i + 1] - 2 * solution[i + 2]) ** 4
                result += 10 * (solution[i] - solution[i + 3]) ** 4
            return result
        else:
            # Handle other functions if needed
            raise ValueError("Unsupported function name")
    
    def F6(self, solution, name="powell_singular2"):
        if name == "powell_singular2":
            D = len(solution)
            result = 0
            for i in range(0, D - 2):
                term1 = (solution[i - 1] + 10 * solution[i]) ** 2
                term2 = 5 * (solution[i + 1] - solution[i + 2]) ** 2
                term3 = (solution[i] - 2 * solution[i + 1]) ** 4
                term4 = 10 * (solution[i - 1] - solution[i + 2]) ** 4
                result += term1 + term2 + term3 + term4
            return result
        else:
            # Handle other functions if needed
            raise ValueError("Unsupported function name")

    def F7(self, solution, name="powell_sum"):
        if name == "powell_sum":
            self.name = name
            n = len(solution) 
            # Check the dimension of the input vector
            if n < 1:
                raise ValueError("Dimension of the input vector must be at least 1.")
            
            result = sum(abs(solution[i])**(i+2) for i in range(n))
            return result
        else:
            # Handle other functions if needed
            raise ValueError("Unsupported function name")
    
    def F8(self, solution, name="qing_function"):
        if name == "qing_function":
            D = len(solution)
            result = 0
            for i in range(D):
                result += (solution[i]**2 - (i + 1))**2  # i+1 is used because indexing in Python starts from 0
            return result
        else:
            # Handle other functions if needed
            raise ValueError("Unsupported function name")
    
    def F9(self, solution, name="quartic_function"):
        if name == "quartic_function":
            D = len(solution)
            result = 0
            for i in range(D):
                result += (i + 1) * (solution[i]**4)  # i+1 is used because indexing in Python starts from 0
            result += random.uniform(0, 1)  # Adding a random number between 0 and 1
            return result
        else:
            # Handle other functions if needed
            raise ValueError("Unsupported function name")
        
    def F10(self, solution, name="rastrigin"):
        if name == "rastrigin":
            A = 10
            n = len(solution)
            if n < 1:
                raise ValueError("Dimension of the input vector must be at least 1.")
            sum_term = np.sum(solution**2 - A * np.cos(2 * np.pi * solution))
            result = A * n + sum_term
            
            return result
        else:
            # Handle other functions if needed
            raise ValueError("Unsupported function name")
    
    def F11(self, solution, name="rosenbrock"):
        if name == "rosenbrock":
            n = len(solution)
            
            if n < 1:
                raise ValueError("Dimension of the input vector must be at least 1.")
            
            result = 0
            
            for i in range(n - 1):
                result += 100 * (solution[i + 1] - solution[i]**2)**2 + (1 - solution[i])**2
            
            return result
        else:
            # Handle other functions if needed
            raise ValueError("Unsupported function name")
    
    def F12(self, solution, name="salomon"):
        if name == "salomon":
            solution = np.array(solution)  # Ensuring solution is a numpy array for vectorized operations
            sum_of_squares = np.sum(solution**2)
            return 1 - np.cos(2 * np.pi * np.sqrt(sum_of_squares)) + 0.1 * np.sqrt(sum_of_squares)
        else:
            # Handle other functions if needed
            raise ValueError("Unsupported function name")
     
    def F13(self, solution, name="schwefel", alpha=0.5):
        if name == "schwefel":
            return sum(xi**2 * alpha for xi in solution)
        else:
            raise ValueError("Unsupported function name")
        
    def F14(self, solution, name="schwefel_1_2"):
        if name == "schwefel_1_2":
            D = len(solution)
            total = 0
            for i in range(D):
                for j in range(D):
                    total += (solution[i] - solution[j])**2
            return total
        else:
            raise ValueError("Unsupported function name")
             
    def F15(self, solution, name="schwefel_2_20"):
        if name == "schwefel_2_20":
            return -sum(abs(xi) for xi in solution)
        else:
            raise ValueError("Unsupported function name")
    
    def F16(self, solution, name="sphere"):
        if name == "sphere":
            n = len(solution)
            
            if n < 1:
                raise ValueError("Dimension of the input vector must be at least 1.")
            
            result = np.sum(solution**2)
        
            return result
        else:
            raise ValueError("Unsupported function name")
    
    def F17(self, solution, name="stepint"):
        if name == "stepint":
            D = len(solution)
            sum_term = sum(int(xi) for xi in solution)
            return 25 + sum_term
        else:
            raise ValueError("Invalid function name")
            
    def F18(self, solution, name="sum_squares"):
        if name == "sum_squares":
            D = len(solution)
            sum_term = sum((i + 1) * xi**2 for i, xi in enumerate(solution))
            return sum_term
        else:
            raise ValueError("Invalid function name")

    # def F19(self, solution, name="weierstrass"):
    #     if name == "weierstrass":
    #         D = len(solution)
    #         a = 0.5
    #         b = 3.0
    #         kmax = 20
    #         result = 0
    #         for i in range(D):
    #             term1 = 0
    #             term2 = 0
    #             for k in range(kmax + 1):
    #                 term1 += a ** k * np.cos(2 * np.pi * b ** k * (solution[i] + 0.5))
    #                 term2 += a ** k * np.cos(np.pi * b ** k)
    #             result += term1 - term2
    #         return result
    #     else:
    #         raise ValueError("Unsupported function name")
        
    def F19(self, solution, name="zakharov"):
        if name == "zakharov":
            sum1 = sum(xi**2 for xi in solution)
            sum2 = sum(i * xi for i, xi in enumerate(solution, start=1))
            return sum1 + sum2**2
        else:
            raise ValueError("Invalid function name")
        


    
            