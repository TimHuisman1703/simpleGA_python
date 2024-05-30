import numpy as np
import time
from functools import partial

import Variation
import Selection
import Offspring
from FitnessFunction import FitnessFunction
from Individual import Individual
from RunStats import RunStats
from Utils import ValueToReachFoundException


class ExtendedCompactGeneticAlgorithm:
    def __init__(self, fitness: FitnessFunction, population_size, **options):
        self.fitness = fitness
        self.evaluation_budget = 1000000
        self.population_size = population_size
        self.population = []
        self.number_of_generations = 0
        self.verbose = False
        self.print_final_results = True
        self.statistics = RunStats()
        self.should_save_stats = False
        self.model = [(j,) for j in range(self.fitness.dimensionality)]

        if "verbose" in options:
            self.verbose = options["verbose"]

        if "evaluation_budget" in options:
            self.evaluation_budget = options["evaluation_budget"]

        if "save_stats" in options:
            self.should_save_stats = options["save_stats"]

    def initialize_population(self):
        self.population = [Individual.initialize_uniform_at_random(self.fitness.dimensionality) for i in
                           range(self.population_size)]
        for individual in self.population:
            self.fitness.evaluate(individual)

    def get_distributions(self):
        distributions = {subset: {} for subset in self.model}

        for individual in self.population:
            for subset in self.model:
                value = tuple(individual.genotype[j] for j in subset)
                distributions[subset][value] = distributions[subset].get(value, 0) + 1

        for subset in self.model:
            s = 0
            options = []
            for key, value in distributions[subset].items():
                s += value
                options.append((key, s / len(self.population)))
            distributions[subset] = options

        return distributions

    def sample_from_distribution(self, distribution):
        u = np.random.random()

        a = 0
        b = len(distribution) - 1
        while a != b:
            mid = (a + b) // 2
            if distribution[mid][1] < u:
                a = mid + 1
            else:
                b = mid

        return distribution[a][0]

    def make_offspring(self):
        distributions = self.get_distributions()

        offspring = []
        for _ in range(len(self.population)):
            new_genotype = [0 for _ in range(self.fitness.dimensionality)]

            for subset in self.model:
                values = self.sample_from_distribution(distributions[subset])
                for key, value in zip(subset, values):
                    new_genotype[key] = value

            new_individual = Individual(new_genotype)
            offspring.append(new_individual)

        for individual in offspring:
            self.fitness.evaluate(individual)

        return offspring

    def make_selection(self, offspring):
        return Selection.tournament_selection(self.population, offspring)

    def print_statistics(self):
        fitness_list = [ind.fitness for ind in self.population]
        print("Generation {}: Best_fitness: {:.1f}, Avg._fitness: {:.3f}, Nr._of_evaluations: {}".format(
            self.number_of_generations, max(fitness_list), np.mean(fitness_list), self.fitness.number_of_evaluations))

    def get_best_fitness(self):
        return max([ind.fitness for ind in self.population])

    def save_stats(self):
        if self.should_save_stats:
            self.statistics.add_generation_stats(self.get_best_fitness(),
                                                 self.get_best_fitness())

    def save_stats_optimal_reached(self):
        if self.should_save_stats:
            self.statistics.add_generation_stats(self.fitness.value_to_reach,
                                                 np.mean([ind.fitness for ind in self.population]))

    def run(self):
        try:
            self.initialize_population()
            while (self.fitness.number_of_evaluations < self.evaluation_budget):
                self.number_of_generations += 1
                if (self.verbose and self.number_of_generations % 100 == 0):
                    self.print_statistics()

                offspring = self.make_offspring()
                selection = self.make_selection(offspring)
                self.population = selection
                self.save_stats()
            if (self.verbose):
                self.print_statistics()
        except ValueToReachFoundException as exception:
            self.save_stats_optimal_reached()
            if (self.print_final_results):

                print(exception)
                print("Best fitness: {:.1f}, Nr._of_evaluations: {}".format(exception.individual.fitness,
                                                                            self.fitness.number_of_evaluations))
            return exception.individual.fitness, self.fitness.number_of_evaluations
        if (self.print_final_results):
            self.print_statistics()
        return self.get_best_fitness(), self.fitness.number_of_evaluations

