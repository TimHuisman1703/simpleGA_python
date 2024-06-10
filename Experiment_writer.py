import os
import pickle
import time

class ExperimentSetup:
    def __init__(self, crossover, local_search, mutation, selection, offspring, population_size, max_budget, set_name, number_of_vertices, instance, times_repeat,  additional_data):
        self.offspring = offspring
        self.selection = selection
        self.crossover = crossover
        self.local_search = local_search
        self.mutation = mutation
        self.population_size = population_size
        self.max_budget = max_budget
        self.set_name = set_name
        self.number_of_vertices = number_of_vertices
        self.instance = instance
        self.additional_data = additional_data
        self.times_repeat = times_repeat


class ExperimentData:
    # Set parameters that should not be added as None
    def __init__(self, crossover, local_search, mutation, selection, offspring, population_size, max_budget, set_name, number_of_vertices, instance, final_value, is_optimal, budget_used, time_taken, additional_data, run_batch_id):
        self.offspring = offspring
        self.selection = selection
        self.crossover = crossover
        self.local_search = local_search # Boolean specifying if local search was enabled
        self.mutation = mutation
        self.population_size = population_size
        self.max_budget = max_budget
        self.set_name = set_name
        self.number_of_vertices = number_of_vertices
        self.instance = instance
        self.final_value = final_value # best found value
        self.is_optimal = is_optimal
        self.budget_used = budget_used
        self.time_taken = time_taken # Milliseconds to run
        self.additional_data = additional_data # Anything additional that might be needed to specify
        self.run_batch_id = run_batch_id # Randomly generated value to group instances made during the same run of the whole program
        self.time_created = int(time.time())

    def save_run(self):
        if not os.path.exists(f"runs"):
            os.makedirs(f"runs")
        with open("runs/all_runs.log", 'ab') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_runs():
        runs = []
        with open("runs/all_runs.log", 'rb') as file:
            while True:
                try:
                    runs.append(pickle.load(file))
                    print("Run written")
                except EOFError:
                    print("Ooopsi, Daisy! Reading the file did not work for some reason. That's sad!")
                    break
        return runs

    @staticmethod
    def remember_locally_done_batch(unique_batch_id):

        if not os.path.exists(f"runs"):
            os.makedirs(f"runs")
        with open("runs/local_batch_ids.txt", 'a') as file:
            file.write(f'{unique_batch_id}\n')