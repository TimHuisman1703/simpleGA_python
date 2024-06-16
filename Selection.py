import numpy as np

def select_best_solution( candidates ):
	best_ind = np.argmax([ind.fitness for ind in candidates])
	return candidates[best_ind]

def tournament_selection( population, offspring ): 
	selection_pool = np.concatenate((population, offspring),axis=None)
	tournament_size = 4
	assert len(selection_pool) % tournament_size == 0, "Population size should be a multiple of 2"
	
	selection = []
	number_of_rounds = tournament_size//2
	for i in range(number_of_rounds):
		number_of_tournaments = len(selection_pool)//tournament_size
		order = np.random.permutation(len(selection_pool))
		for j in range(number_of_tournaments):
			indices = order[tournament_size*j:tournament_size*(j+1)]
			best = select_best_solution(selection_pool[indices])
			selection.append(best)
	assert( len(selection) == len(population) )

	return selection


def best_solutions_only(population, offspring):
	combined = population + offspring
	k = len(population)
	return sorted(combined, key=lambda x: x.fitness, reverse=True)[:k]

def fitness_sharing(population, alpha=1):
    def calculate_shared_fitness(individual: Individual, population):
        niche_count = 0
        for other_individual in population:
            distance = np.sum(individual.genotype != other_individual.genotype)
            if distance < 1:
                niche_count += 1 - (distance / 1)**alpha
        return individual.fitness / niche_count

    for individual in population:
        individual.shared_fitness = calculate_shared_fitness(individual, population)

    return sorted(population, key=lambda x: x.shared_fitness, reverse=True)

def fitness_sharing_selection(population, offspring, selection_size=4):
    combined_population = population + offspring
    shared_population = fitness_sharing(combined_population)
    return shared_population[:selection_size]


