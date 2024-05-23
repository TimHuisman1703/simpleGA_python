class RunStats:
    def __init__(self):
        self.best_fitness_per_generation = []
        self.average_fitness_per_generation = []

    def add_generation_stats(self, best_fitness, average_fitness):
        self.best_fitness_per_generation.append(best_fitness)
        self.average_fitness_per_generation.append(average_fitness)
