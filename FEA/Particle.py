import numpy as np
import random
from operator import attrgetter
from copy import deepcopy, copy

# ANSI escape codes for colors
RED = "\033[31m"
BLUE = "\033[34m"
ENDC = "\033[0m"  # Reset to default color

class Particle(object):
    def __init__(self, function, dim, position=None, factor=None, global_solution=None, lbest_pos=None):
        """
        Initializes a particle in the Particle Swarm Optimization (PSO) algorithm.

        :param function: The objective function for optimization.
        :param dim: The number of dimensions of the search space.
        :param position: The initial position of the particle in the search space. If None, a random position is assigned.
        :param factor: The subset of dimensions this particle is responsible for (used in Factored PSO).
        :param global_solution: The reference to the global solution in the swarm.
        :param lbest_pos: The best position the particle has achieved so far (local best position).
        """
        self.f = function  # Objective function
        self.lbest_fitness = float('inf')  # Initialize local best fitness as infinity
        self.dim = dim  # Number of dimensions
        self.factor = factor  # Subset of dimensions assigned to this particle

        # Initialize particle's position and local best position
        if position is None:
            self.position = np.random.uniform(function.lbound, function.ubound, size=dim)
            self.lbest_position = np.array([x for x in self.position])
        else:
            self.position = position
            self.lbest_position = lbest_pos
            self.lbest_fitness = self.calculate_fitness(global_solution, lbest_pos)

        self.velocity = np.zeros(dim)  # Initialize particle's velocity
        self.fitness = self.calculate_fitness(global_solution)  # Calculate current fitness

    # Comparison operators for particles based on their fitness values
    def __le__(self, other): ...
    def __lt__(self, other): ...
    def __gt__(self, other): ...
    def __eq__(self, other): ...

    def __str__(self):
        """
        String representation of the particle, showing its current and best fitness.
        """
        return ' '.join([RED + 'Particle with current fitness:', str(self.fitness) + ENDC, BLUE + 'and best fitness:', str(self.lbest_fitness) + ENDC])

    def set_fitness(self, fit):
        """
        Sets the fitness of the particle and updates local best fitness and position if necessary.
        """
        self.fitness = fit
        if fit < self.lbest_fitness:
            self.lbest_fitness = deepcopy(fit)
            self.lbest_position = np.array([x for x in self.position])

    def set_position(self, position):
        """
        Sets the position of the particle in the search space.
        """
        self.position = np.array(position)

    def update_individual_after_compete(self, global_solution=None):
        """
        Updates the particle after competing with other particles, particularly in the context of FEA.
        """
        fitness = self.calculate_fitness(global_solution)
        if fitness < self.lbest_fitness:
            self.lbest_fitness = deepcopy(fitness)
        self.fitness = fitness
        return self

    def calculate_fitness(self, glob_solution, position=None):
        """
        Calculates the fitness of the particle, either in isolation or as part of a global solution.
        """
        # Fitness calculation is adjusted based on whether the particle is part of a global solution or not
        if glob_solution is None:
            fitness = self.f.run(self.position)
        else:
            solution = [x for x in glob_solution]
            if position is None:
                for i, x in zip(self.factor, self.position):
                    solution[i] = x
            else:
                for i, x in zip(self.factor, position):
                    solution[i] = x
            fitness = self.f.run(np.array(solution))
        return fitness

    def update_particle(self, omega, phi, global_best_position, v_max, global_solution=None):
        """
        Updates the particle's position and velocity based on PSO dynamics.
        """
        self.update_velocity(omega, phi, global_best_position, v_max)
        self.update_position(global_solution)

    def update_velocity(self, omega, phi, global_best_position, v_max):
        """
        Updates the velocity of the particle based on inertia, cognitive, and social components.
        """
        # Calculating new velocity by combining inertia, personal (cognitive), and social components
        inertia = np.multiply(omega, self.velocity)
        phi_1 = np.array([random.random() * phi for _ in range(self.dim)])  # Exploration term
        personal_exploitation = self.lbest_position - self.position  # Cognitive component
        phi_2 = np.array([random.random() * phi for _ in range(self.dim)])  # Exploration term
        social_exploitation = global_best_position - self.position  # Social component
        new_velocity = inertia + phi_1 * personal_exploitation + phi_2 * social_exploitation
        self.velocity = np.array([self.clamp_value(v, -v_max, v_max) for v in new_velocity])  # Clamping the velocity

    def update_position(self, global_solution=None):
        """
        Updates the position of the particle in the search space.
        """
        # Update position based on velocity and clamp it within bounds
        lo, hi = self.f.lbound, self.f.ubound
        position = self.velocity + self.position
        self.position = np.array([self.clamp_value(p, lo, hi) for p in position])
        self.fitness = self.calculate_fitness(global_solution)  # Update fitness

    def clamp_value(self, value, lo, hi):
        """
        Clamps a given value within specified lower and upper bounds.
        """
        if lo <= value <= hi:
            return value
        elif value < lo:
            return lo
        return hi
