import numpy as np
from Utils import hamming_distance


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
            distances[i][j] = hamming_distance(population[i].genotype, population[j].genotype)
    d_hat = sum(sum(d) for d in distances)
    d_hat = 2 * d_hat / (len(population) * (len(population) - 1))

    order = np.random.permutation(len(population))
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            p_a = population[order[i]]
            p_b = population[order[j]]
            if distances[order[i]][order[j]] >= d_hat:  # >= instead of > to avoid problems when all parents are the same
                return [variation_operator(p_a, p_b)]
    raise ValueError("No suitable parents found")
