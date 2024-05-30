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


class GeneticAlgorithm:
    def __init__(self, fitness: FitnessFunction, population_size, **options):
        self.fitness = fitness
        self.evaluation_budget = 1000000
        self.variation_operator = Variation.uniform_crossover
        self.selection_operator = Selection.tournament_selection
        self.offspring_operator = Offspring.default
        self.population_size = population_size
        self.population = []
        self.number_of_generations = 0
        self.verbose = False
        self.print_final_results = True
        self.statistics = RunStats()
        self.should_save_stats = False

        if "verbose" in options:
            self.verbose = options["verbose"]

        if "evaluation_budget" in options:
            self.evaluation_budget = options["evaluation_budget"]

        if "variation" in options:
            if options["variation"] == "UniformCrossover":
                self.variation_operator = Variation.uniform_crossover
            elif options["variation"] == "OnePointCrossover":
                self.variation_operator = Variation.one_point_crossover
            elif options["variation"] == "TwoPointCrossover":
                self.variation_operator = Variation.two_point_crossover
            elif options["variation"] == "GreedyCrossover":
                self.variation_operator = partial(Variation.greedy_crossover, self.fitness)
            elif options["variation"] == "GreedyMutCrossover":
                self.variation_operator = partial(Variation.greedy_crossover_with_mutation, self.fitness)
            elif options['variation'] == 'Qinghua':
                self.variation_operator = partial(Variation.qinghua_operator, self.fitness)
            # elif options["variation"] == "CustomCrossover":
            #     self.variation_operator = partial(Variation.custom_crossover, self.fitness)

        if 'selection' in options:
            if options['selection'] == 'TournamentSelection':
                self.selection_operator = Selection.tournament_selection
            elif options['selection'] == 'BestSolutionsOnly':
                self.selection_operator = Selection.best_solutions_only

        if 'offspring' in options:
            if options['offspring'] == 'Default':
                self.offspring_operator = Offspring.default
            elif options['offspring'] == 'Qinghua':
                self.offspring_operator = Offspring.qinghua_operator

        if "save_stats" in options:
            self.should_save_stats = options["save_stats"]

    def initialize_population(self):
        self.population = [Individual.initialize_uniform_at_random(self.fitness.dimensionality) for i in
                           range(self.population_size)]
        for individual in self.population:
            self.fitness.evaluate(individual)

    def make_offspring(self):
        offspring = self.offspring_operator(self.population, self.variation_operator)
        for individual in offspring:
            self.fitness.evaluate(individual)
            # print(individual.fitness)
        return offspring

    def make_selection(self, offspring):
        return self.selection_operator(self.population, offspring)

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

