import numpy as np
from Utils import hamming_distance
import FitnessFunction
import Individual

def default(population, variation_operator):
    offspring = []
    order = np.random.permutation(len(population))
    for i in range(len(order) // 2):
        offspring = offspring + variation_operator(population[order[2 * i]],
                                                   population[order[2 * i + 1]])
    return offspring


def qinghua_operator(population, variation_operator):
    d_hat = 0.0  # the average hamming distance between the parents
    distances = [[0for _ in range(len(population))] for _ in range(len(population))]
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distance = hamming_distance(population[i].genotype, population[j].genotype)
            distances[i][j] = distance
            distances[j][i] = distance
    d_hat = sum(sum(d) for d in distances)
    d_hat = d_hat / len(population) ** 2

    order = np.random.permutation(len(population))
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            p_a = population[order[i]]
            p_b = population[order[j]]
            if distances[order[i]][order[j]] >= d_hat:  # >= instead of > to avoid problems when all parents are the same
                return [variation_operator(p_a, p_b)]
    raise ValueError("No suitable parents found")

def simulated_annealing(fitness: FitnessFunction, individual: Individual, max_iterations=100, initial_temp=100, cooling_rate=0.99):
    current_genotype = individual.genotype.copy()
    current_fitness = individual.fitness
    temp = initial_temp

    for _ in range(max_iterations):
        if temp <= 0:
            break
        new_genotype = current_genotype.copy()
        i = np.random.randint(len(new_genotype))
        new_genotype[i] = 1 - new_genotype[i]
        new_fitness = fitness.partial_evaluate(current_genotype, i, new_genotype[i])
            
        if new_fitness > current_fitness or np.random.rand() < np.exp((new_fitness - current_fitness) / temp):
            current_genotype = new_genotype
            current_fitness = new_fitness
            
        temp *= cooling_rate
        
    individual.genotype = current_genotype
    individual.fitness = current_fitness
    return individual

def simulated_annealing_wrapper(fitness, individual):
    return [simulated_annealing(fitness, individual)]
