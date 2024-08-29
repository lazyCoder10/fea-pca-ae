import numpy as np
import random
from operator import attrgetter
from copy import deepcopy, copy
from Particle import Particle

# ANSI escape codes for colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
ENDC = "\033[0m"  # Reset to default color

class PSO(object):
    def __init__(self, generations, population_size, function, dim, factor=None, global_solution=None, omega=0.729, phi=1.49618):
        # Constructor method for the Particle Swarm Optimization (PSO) class.
        # Initializes the PSO algorithm with the given parameters.

        # Initialize population size and create a list of particles.
        self.pop_size = population_size
        self.pop = [Particle(function, dim, factor=factor, global_solution=global_solution) for x in range(population_size)]

        # Write the initial positions of particles to a file named 'pso2.o'.
        pos = [p.position for p in self.pop]
        with open('pso2.o', 'a') as file:
            file.write(str(pos))
            file.write('\n')

        # Set PSO parameters and variables.
        self.omega = omega
        self.phi = phi
        self.f = function
        self.dim = dim
        pbest_particle = Particle(function, dim, factor=factor, global_solution=global_solution)
        pbest_particle.set_fitness(float('inf'))
        self.pbest_history = [pbest_particle]
        self.gbest = pbest_particle
        self.v_max = abs((function.ubound - function.lbound))
        self.generations = generations
        self.current_loop = 0
        self.factor = np.array(factor)
        self.global_solution = global_solution

    def find_current_best(self):
        # Find the current best particle (lowest fitness) in the population.
        sorted_ = sorted(np.array(self.pop), key=attrgetter('fitness'))
        return Particle(self.f, self.dim, position=sorted_[0].position, factor=self.factor,
                 global_solution=self.global_solution, lbest_pos=sorted_[0].lbest_position)

    def find_local_best(self):
        # Placeholder for finding a local best solution (not implemented).
        pass

    def update_swarm(self):
        # Update the particle swarm for one iteration.

        # Determine global solution to be considered.
        if self.global_solution is not None:
            global_solution = [x for x in self.global_solution]
        else:
            global_solution = None
        
        # Extract PSO parameters.
        omega, phi, v_max = self.omega, self.phi, self.v_max
        global_best_position = [x for x in self.gbest.position]

        # Update each particle in the population.
        for p in self.pop:
            p.update_particle(omega, phi, global_best_position, v_max, global_solution)

        # Find the current best particle and update the global best if necessary.
        curr_best = self.find_current_best()
        self.pbest_history.append(curr_best)
        if curr_best.fitness < self.gbest.fitness:
            self.gbest = curr_best

    def replace_worst_solution(self, global_solution):
        # Replace the worst particle in the population with a new solution.

        # Convert the global solution to a numpy array.
        self.global_solution = np.array([x for x in global_solution])

        # Sort the population by fitness and print information.
        self.pop.sort(key=attrgetter('fitness'))
        # print(YELLOW + f"Worst Fitness: {self.pop[-1].fitness}, Best Fitness: {self.pop[0].fitness}" + ENDC)
        # print(self.pop[-1], self.pop[0])

        # Create a partial solution based on the specified factor.
        partial_solution = [x for i, x in enumerate(global_solution) if i in self.factor]

        # Set the position and fitness of the worst particle to the new solution.
        self.pop[-1].set_position(partial_solution)
        self.pop[-1].set_fitness(self.f.run(self.global_solution))

        # Create a new current best particle and shuffle the population.
        curr_best = Particle(self.f, self.dim, position=self.pop[0].position, factor=self.factor,
                 global_solution=self.global_solution, lbest_pos=self.pop[0].lbest_position)
        random.shuffle(self.pop)

        # Update the global best if necessary.
        if curr_best.fitness < self.gbest.fitness:
            self.gbest = curr_best

    def run(self):
        # Run the PSO algorithm for the specified number of generations.

        for i in range(self.generations):
            self.update_swarm()
            self.current_loop += 1
            # print(self.gbest)
        
        # Return the position of the global best solution.
        return self.gbest.position
