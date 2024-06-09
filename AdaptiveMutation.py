import numpy as np
import FitnessFunction
from Individual import Individual

rate = 0.05
MAX_RATE = 0.10
MIN_RATE = 0.001
DIVERSITY_THRESHOLD = 0.1
        
def calculate_hamming_distance(genotype1, genotype2):
    return np.sum(np.array(genotype1) != np.array(genotype2))

def calculate_diversity(population):
    n = len(population)
    if n <= 1:
        return 0
    hamming_distances = []
    for i in range(n):
        for j in range(i + 1, n):
            hamming_distances.append(calculate_hamming_distance(population[i].genotype, population[j].genotype))
    average_hamming_distance = np.mean(hamming_distances)
    return average_hamming_distance / len(population[0].genotype)  # Normalize by genotype length

def adjust_rate(diversity):
    global rate
    if diversity < DIVERSITY_THRESHOLD:
        rate = min(rate * 1.1, MAX_RATE)
    else:
        rate = max(rate * 0.9, MIN_RATE)

def mutate(individual):
    new_genotype = individual.genotype.copy()
    for i in range(len(new_genotype)):
        if np.random.uniform(0.0, 1.0) < rate:
            new_genotype[i] = 1 - new_genotype[i]
    individual.genotype = new_genotype
    return individual

def adaptive_mutation_wrapper(individual: Individual, population, fitness: FitnessFunction):
    diversity = calculate_diversity(population)
    adjust_rate(diversity)
    mutated_individual = mutate(individual)
    fitness.evaluate(mutated_individual)
    return mutated_individual

    


