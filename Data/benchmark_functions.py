import numpy as np
import math
import random

# function 1
def ackley(x):
    # Define the constants a, b, and c for the Ackley function
    a = 20
    b = 0.2
    c = 2 * math.pi
    # Get the number of dimensions (n) from the length of the input array x
    n = len(x)
    # Calculate the first part of the Ackley function
    sum1 = np.sum(x ** 2)
    # Calculate the second part of the Ackley function
    sum2 = np.sum(np.cos(c * x))
    # Calculate the individual terms for the Ackley function
    term1 = -a * math.exp(-b * math.sqrt(sum1 / n))
    term2 = -math.exp(sum2 / n)
    # Combine the terms and add constants to get the final result
    result = term1 + term2 + a + math.exp(1)
    
    return result

# # function 2
# def brown(x):
#     # Extract the elements of the input vector x except the last one (xi)
#     xi = x[:-1]
#     # Extract the elements of the input vector x starting from the second element (xi+1)
#     xi1 = x[1:]
#     # Calculate the terms for each pair of xi and xi+1
#     with np.errstate(over='ignore'):
#         terms = ((xi ** 2) ** ((xi ** 2 + xi1 + 1) / (xi ** 2 + xi1)))
#     # Sum up the terms and add the first element of the input vector (x0)
#     result = terms.sum() + x[0]
#     # Return the final result
#     return result

def dixon_price(x):
    n = len(x) 
    # Check the dimension of the input vector
    if n < 1:
        raise ValueError("Dimension of the input vector must be at least 1.")
    result = (x[0] - 1) ** 2
    for i in range(1, n):
        result += (i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2
    
    return result


# The Griewank function has many widespread local minima, which are regularly distributed.
def griewank(x):
    n = len(x)
    sum_sq = np.sum(np.square(x))
    prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    result = 1 + (sum_sq / 4000) - prod_cos
    return result

def powell_singular(x):
    n = len(x)
    if n < 4:
        raise ValueError("Dimension of the input vector must be at least 4.")
    result = 0
    for i in range(0, n - 3, 4):
        result += (x[i] + 10 * x[i + 1]) ** 2
        result += 5 * (x[i + 2] - x[i + 3]) ** 2
        result += (x[i + 1] - 2 * x[i + 2]) ** 4
        result += 10 * (x[i] - x[i + 3]) ** 4
    return result

def powell_singular2 (x):
    D = len(x)
    result = 0
    for i in range(0, D - 2):
        term1 = (x[i - 1] + 10 * x[i]) ** 2
        term2 = 5 * (x[i + 1] - x[i + 2]) ** 2
        term3 = (x[i] - 2 * x[i + 1]) ** 4
        term4 = 10 * (x[i - 1] - x[i + 2]) ** 4
        result += term1 + term2 + term3 + term4
    return result

def powell_sum(x):
    """
    Compute the Powell Sum function.

    Parameters:
    x (np.array): A numpy array of shape (D, ) representing a D-dimensional vector.

    Returns:
    float: The value of the Powell Sum function evaluated at x.
    """
    D = len(x)
    return sum(abs(x[i])**(i+2) for i in range(D))

def qing_function(x):
    D = len(x)
    result = 0
    for i in range(D):
        result += (x[i]**2 - (i + 1))**2  # i+1 is used because indexing in Python starts from 0
    return result

def quartic_function(x):
    D = len(x)
    result = 0
    for i in range(D):
        result += (i + 1) * (x[i]**4)  # i+1 is used because indexing in Python starts from 0
    result += random.uniform(0, 1)  # Adding a random number between 0 and 1
    return result

# function 8
def rastrigin(x):
    A = 10
    n = len(x)
    if n < 1:
        raise ValueError("Dimension of the input vector must be at least 1.")
    sum_term = np.sum(x**2 - A * np.cos(2 * np.pi * x))
    result = A * n + sum_term
    
    return result

# function 9
def rosenbrock(x):
    n = len(x)
    
    if n < 1:
        raise ValueError("Dimension of the input vector must be at least 1.")
    
    result = 0
    
    for i in range(n - 1):
        result += 100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2
    
    return result


def salomon(x):
    x = np.array(x)  # Ensuring x is a numpy array for vectorized operations
    sum_of_squares = np.sum(x**2)
    return 1 - np.cos(2 * np.pi * np.sqrt(sum_of_squares)) + 0.1 * np.sqrt(sum_of_squares)


def schumer_steiglitz(x):
    D = len(x)
    result = 0
    for i in range(D):
        result += x[i] ** (4 * (i + 1))
    return result


# function 11
def sphere(x):
    n = len(x)
    
    if n < 1:
        raise ValueError("Dimension of the input vector must be at least 1.")
    
    result = np.sum(x**2)
    
    return result


def schwefel(x, alpha=0.5):
    """
    Schwefel Function

    Args:
    - x (list or numpy.ndarray): A list or an array of coordinates.
    - alpha (float): The exponent parameter, typically 0.5.

    Returns:
    - float: The function's value at x.

    Description:
    This is a continuous, differentiable, partially-separable, scalable, unimodal function.
    It is defined as the sum of the individual elements of x raised to the power of 2*alpha.
    The function is defined for x values in the range of [-100, 100].

    The global minimum is located at x* = (0, ..., 0), and f(x*) = 0.
    """

    return sum(xi**2 * alpha for xi in x)

def schwefel_1_2(x):
    """
    Schwefel 1.2 Function

    Args:
    - x (list or numpy.ndarray): A list or an array of coordinates.

    Returns:
    - float: The function's value at x.

    Description:
    This is a continuous, differentiable, non-separable, scalable, unimodal function.
    It is defined as the sum of the squared differences of all pairs of elements in x.
    The function is defined for x values in the range of [-100, 100].

    The global minimum is located at x* = (0, ..., 0), and f(x*) = 0.
    """

    D = len(x)
    total = 0
    for i in range(D):
        for j in range(D):
            total += (x[i] - x[j])**2

    return total

def schwefel_2_20(x):
    """
    Schwefel 2.20 Function

    Args:
    - x (list or numpy.ndarray): A list or an array of coordinates.

    Returns:
    - float: The function's value at x.

    Description:
    This is a continuous, non-differentiable, separable, scalable, unimodal function.
    It is defined as the negative sum of the absolute values of the elements in x.
    The function is defined for x values in the range of [-100, 100].

    The global minimum is located at x* = (0, ..., 0), and f(x*) = 0.
    """

    return -sum(abs(xi) for xi in x)


def streched_v_sine_wave(x):
    D = len(x)
    result = 0
    for i in range(D):
        term1 = x[2 * i] ** 2 + x[2 * i + 1] ** 2
        term2 = np.sin(50 * term1 ** 0.1) ** 2
        result += term1 ** 0.25 * term2 + 0.1 * i
    return result

def weierstrass(x):
    D = len(x)
    a = 0.5
    b = 3.0
    kmax = 20
    result = 0
    for i in range(D):
        term1 = 0
        term2 = 0
        for k in range(kmax + 1):
            term1 += a ** k * np.cos(2 * np.pi * b ** k * (x[i] + 0.5))
            term2 += a ** k * np.cos(np.pi * b ** k)
        result += term1 - term2
    return result

def brown(x):
    """
    Computes the Brown function value for a given input vector x.

    Args:
    x (list or numpy array): Input vector.

    Returns:
    float: The function value at x.
    """
    n = len(x)
    sum = 0
    for i in range(n - 1):
        sum += (x[i]**2)**(x[i + 1]**2 + 1) + (x[i + 1]**2)**(x[i]**2 + 1)
    return sum

def zakharov(x):
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(i * xi for i, xi in enumerate(x, start=1))
    return sum1 + sum2**2

def stepint(x):
    D = len(x)
    sum_term = sum(int(xi) for xi in x)
    return 25 + sum_term

def sum_squares(x):
    D = len(x)
    sum_term = sum((i + 1) * xi**2 for i, xi in enumerate(x))
    return sum_term


def ackley__(z):
    """
    Ackley function implementation for a given input vector z.
    """
    n = len(z)
    sum1 = np.sum(z ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * z))

    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)

    return term1 + term2 + 20 + np.e


def F11(solution, shift_data, rotation_matrix, m_group=150):
    """
    Compute the D_2m-group shifted and m-rotated Ackley's function.

    Parameters:
    - solution: Input vector.
    - shift_data: The shift data (vector).
    - rotation_matrix: The rotation matrix.
    - m_group: The group size for shifting and rotating.

    Returns:
    - Function value.
    """
    dimensions = len(solution)
    epoch = int(dimensions / (2 * m_group))

    # Ensure the shift_data and rotation_matrix are appropriately sized
    shift_data = shift_data[:dimensions]
    permutation_data = np.random.permutation(dimensions)

    z = solution - shift_data
    result = 0.0

    for i in range(0, epoch):
        idx1 = permutation_data[i * m_group:(i + 1) * m_group]
        z1 = np.dot(z[idx1], rotation_matrix[:len(idx1), :len(idx1)])
        result += ackley__(z1)

    idx2 = permutation_data[int(dimensions / 2):dimensions]
    z2 = z[idx2]
    result += ackley__(z2)

    return result