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

    def get_distribution_subset(self, subset):
        distribution = {}

        for individual in self.population:
            value = tuple(individual.genotype[j] for j in subset)
            distribution[value] = distribution.get(value, 0) + 1 / len(self.population)

        return distribution

    def get_distributions(self, model):
        return {subset: self.get_distribution_subset(subset) for subset in model}

    def sample_from_distribution(self, distribution):
        u = np.random.random()

        s = 0
        for key, value in distribution.items():
            s += value
            if u <= s:
                return key

    def get_complexity_of_subset(self, subset):
        distribution = self.get_distribution_subset(subset)

        model_complexity = ((1 << len(subset)) - 1) * np.log2(len(self.population) + 1)

        compressed_population_complexity = 0.0
        for p in distribution.values():
            if p > 0.0:
                compressed_population_complexity += -p * np.log2(p)
        compressed_population_complexity *= len(self.population)

        return model_complexity + compressed_population_complexity

    def mutate_model(self, model):
        best_model = model
        best_model_complexity_diff = 0.0

        # Merge two subsets
        for i in range(len(model)):
            for j in range(i + 1, len(model)):
                complexity_added = self.get_complexity_of_subset(tuple(sorted([*model[i], *model[j]])))
                complexity_removed = self.get_complexity_of_subset(model[i]) + self.get_complexity_of_subset(model[j])
                complexity_diff = complexity_added - complexity_removed
                if complexity_diff < best_model_complexity_diff:
                    best_model = model[:i] + model[i+1:j] + model[j+1:] + [tuple(sorted([*model[i], *model[j]]))]

        if best_model == model:
            return best_model
        else:
            return self.mutate_model(best_model)

    def make_offspring(self):
        distributions = self.get_distributions(self.model)

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
        print("\033[31mGeneration {}\033[0m: Best_fitness: {:.1f}, Avg._fitness: {:.3f}, Nr._of_evaluations: {}".format(
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

                self.model = self.mutate_model(self.model)
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
