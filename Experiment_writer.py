import os
import pickle
import time
from itertools import groupby
from operator import itemgetter
from collections import defaultdict

def create_nested_dict(n):
    if n <= 1:
        return {}
    else:
        return defaultdict(lambda: create_nested_dict(n - 1))

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
    def __init__(self, crossover, local_search, mutation, selection, offspring, population_size, max_budget, set_name, number_of_vertices, instance, final_value, is_optimal, budget_used, time_taken, additional_data, run_batch_id, has_converged):
        self.has_converged = has_converged
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
        with open("runs/all_runs_2.log", 'ab') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_runs():
        ExperimentData.load_runs_with_name("all_runs")

    @staticmethod
    def load_runs_with_name(name):
        runs = []
        with open(f"runs/{name}.log", 'rb') as file:
            while True:
                try:
                    runs.append(pickle.load(file))
                    #print("Run read")
                except EOFError:
                    print(f"All the runs have been read for {name}")
                    break
        return runs

    @staticmethod
    def remember_locally_done_batch(unique_batch_id):

        if not os.path.exists(f"runs"):
            os.makedirs(f"runs")
        with open("runs/local_batch_ids.txt", 'a') as file:
            file.write(f'{unique_batch_id}\n')

    @staticmethod
    def group_same_executions(exp_data):
        data_dicts = [data_point.__dict__ for data_point in exp_data]
        grouped_objects = groupby(data_dicts, key=itemgetter('offspring', 'selection', 'crossover', 'local_search', 'mutation', 'population_size', 'max_budget', 'set_name', 'instance' ))
        # dict instead because groupby ignores keys with None
        return {key: list(group) for key, group in grouped_objects}

    @staticmethod
    def find_average_values_in_grouped(grouped_data):
        averaged_run_data_dict = {}
        for key in grouped_data:
            values = grouped_data[key]
            group_list = list(values)
            total_experiments = len(group_list)
            avg_has_converged = sum(obj['has_converged'] for obj in group_list) / len(group_list)
            avg_final_value = sum(obj['final_value'] for obj in group_list) / len(group_list)
            times_opt_found = sum(obj['is_optimal'] for obj in group_list)
            avg_budget_used = sum(obj['budget_used'] for obj in group_list) / len(group_list)
            avg_time_taken = sum(obj['time_taken'] for obj in group_list) / len(group_list)
            averaged_run_data_dict[key] = {
                "total_experiments" : total_experiments,
                "avg_has_converged" : avg_has_converged,
                "avg_final_value" : avg_final_value,
                "times_opt_found" : times_opt_found,
                "avg_budget_used" : avg_budget_used,
                "avg_time_taken": avg_time_taken,
            }

        return averaged_run_data_dict

    @staticmethod
    def group_same_sets_then_instances(averaged_run_data):
        nested_dict = create_nested_dict(5)

        # Populate the nested dictionary
        for key_tuple, value in averaged_run_data.items():
            offspring, selection, crossover, local_search, mutation, population_size, max_budget, set_name, instance = key_tuple
            nested_dict[set_name][instance][local_search][crossover][(offspring, selection, mutation, population_size, max_budget)] = value
        return  nested_dict

    @staticmethod
    def load_multiple_runs(names_of_runs):
        concatenated_result = []
        for name in names_of_runs:
            concatenated_result.extend(ExperimentData.load_runs_with_name(name))
        return concatenated_result



