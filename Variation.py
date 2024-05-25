import numpy as np

from Individual import Individual
from FitnessFunction import FitnessFunction

MUTATION_CHANCE = 0.05
OPTIMAL_CROSSOVER_CHANCE = 0.9

def uniform_crossover(individual_a: Individual, individual_b: Individual, p = 0.5 ):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(l)
    offspring_b = Individual(l)
    
    m = np.random.choice((0,1), p=(p, 1-p), size=l)
    offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
    offspring_b.genotype = np.where(1 - m, individual_a.genotype, individual_b.genotype)
    
    return [offspring_a, offspring_b]

def one_point_crossover(individual_a: Individual, individual_b: Individual ):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(l)
    offspring_b = Individual(l)
    
    l = len(individual_a.genotype)
    m = np.arange(l) < np.random.randint(l+1)
    offspring_a.genotype = np.where(m, individual_a.genotype, individual_b.genotype)
    offspring_b.genotype = np.where(~m, individual_a.genotype, individual_b.genotype)
    
    return [offspring_a, offspring_b]

def two_point_crossover(individual_a: Individual, individual_b: Individual ):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    offspring_a = Individual()
    offspring_b = Individual()
    
    l = len(individual_a.genotype)
    m = (np.arange(l) < np.random.randint(l+1)) ^ (np.arange(l) < np.random.randint(l+1))
    offspring_a.genotype = np.where(m, individual_b.genotype, individual_a.genotype)
    offspring_b.genotype = np.where(~m, individual_b.genotype, individual_a.genotype)
    
    return [offspring_a, offspring_b]




def greedy_crossover( fitness: FitnessFunction, individual_a: Individual, individual_b: Individual ):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(np.zeros(l))
    offspring_b = Individual(np.zeros(l))
   
    for i in range(len(individual_a.genotype)):
        c_a = individual_a.genotype[i]
        c_b = individual_b.genotype[i]

        # only pick the most optimal if parents have differing genes
        if c_a != c_b:
            if np.random.uniform(0.0, 1.0) < OPTIMAL_CROSSOVER_CHANCE:
                c_a = fitness.partial(individual_a, i)

            if np.random.uniform(0.0, 1.0) < OPTIMAL_CROSSOVER_CHANCE:
                c_b = fitness.partial(individual_b, i)

        # offspring modeled from individual a
        offspring_a.genotype[i] = c_a
        # offsrping modeled from individual b
        offspring_b.genotype[i] = c_b

    return [offspring_a, offspring_b]


def greedy_crossover_with_mutation( fitness: FitnessFunction, individual_a: Individual, individual_b: Individual ):
    assert len(individual_a.genotype) == len(individual_b.genotype), "solutions should be equal in size"
    l = len(individual_a.genotype)
    offspring_a = Individual(np.zeros(l))
    offspring_b = Individual(np.zeros(l))
   
    for i in range(len(individual_a.genotype)):
        c_a = individual_a.genotype[i]
        c_b = individual_b.genotype[i]

        # only pick the most optimal if parents have differing genes
        if c_a != c_b:
            if np.random.uniform(0.0, 1.0) < OPTIMAL_CROSSOVER_CHANCE:
                c_a = fitness.partial(individual_a, i)

            if np.random.uniform(0.0, 1.0) < OPTIMAL_CROSSOVER_CHANCE:
                c_b = fitness.partial(individual_b, i)

        # apply random mutation 
        if np.random.uniform(0.0,1.0) < MUTATION_CHANCE:
            c_a = 1 - c_a

        if np.random.uniform(0.0,1.0) < MUTATION_CHANCE:
            c_b = 1 - c_b

        # offspring modeled from individual a
        offspring_a.genotype[i] = c_a
        # offsrping modeled from individual b
        offspring_b.genotype[i] = c_b

    return [offspring_a, offspring_b]

