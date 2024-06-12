import os
import pickle
import time
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
from statistics import variance

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
        with open("runs/all_runs_missing.log", 'ab') as file:
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
            var = variance((obj['budget_used'] for obj in group_list))
            averaged_run_data_dict[key] = {
                "total_experiments" : total_experiments,
                "avg_has_converged" : avg_has_converged,
                "avg_final_value" : avg_final_value,
                "times_opt_found" : times_opt_found,
                "avg_budget_used" : avg_budget_used,
                "avg_time_taken": avg_time_taken,
                "variance": var
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

    @staticmethod
    def leave_best_performing_population(grouped):
        nested_dict = create_nested_dict(5)
        for set_name, in_dict in grouped.items():
            for instance, in_dict1 in in_dict.items():
                for local_search, in_dict2 in in_dict1.items():
                    for crossover, in_dict3 in in_dict2.items():
                        smallest_avg_budget = 10000000000000000000000000000
                        best_configuration = None
                        for config, in_dict4 in in_dict3.items():
                            if in_dict4["avg_budget_used"] < smallest_avg_budget:
                                smallest_avg_budget = in_dict4["avg_budget_used"]
                                best_configuration = in_dict4
                                best_configuration["offspring"] = config[0] # (offspring, selection, mutation, population_size, max_budget)
                                best_configuration["selection"] = config[1]
                                best_configuration["mutation"] = config[2]
                                best_configuration["population_size"] = config[3]
                                best_configuration["max_budget"] = config[4]
                        nested_dict[set_name][instance][local_search][crossover] = best_configuration
        return nested_dict

    @staticmethod
    def sort_by_performance(grouped):
        nested_dict = create_nested_dict(2)
        for set_name, in_dict in grouped.items():
            for instance, in_dict1 in in_dict.items():
                performances_for_same_instance = []
                for local_search, in_dict2 in in_dict1.items():
                    for crossover, in_dict3 in in_dict2.items():
                        performances_for_same_instance.append(((crossover, local_search), in_dict3))
                sorted_values = sorted(performances_for_same_instance, key=lambda x: x[1]["avg_budget_used"], reverse=False)
                # print(f"For {instance} in {set_name} the best performing configurations are: ")
                # for config in sorted_values:
                #     print(f"Crossover {config[0][0]} with LS={config[0][1]} pop={config[1]['population_size']} and avg evals at {config[1]['avg_budget_used']}")
                # print("")
                nested_dict[set_name][instance] = sorted_values
        return nested_dict

    @staticmethod
    def sort_by_crossover(grouped):
        nested_dict = create_nested_dict(2)
        for set_name, in_dict in grouped.items():
            for instance, in_dict1 in in_dict.items():
                performances_for_same_instance = []
                for local_search, in_dict2 in in_dict1.items():
                    for crossover, in_dict3 in in_dict2.items():
                        performances_for_same_instance.append(((crossover, local_search), in_dict3))
                sorted_values_by_name = sorted(performances_for_same_instance, key=lambda x: (x[0][0]+str(x[0][1])), reverse=False)
                nested_dict[set_name][instance] = sorted_values_by_name
        return nested_dict

    @staticmethod
    def remove_small_test_size(grouped):
        correct_size = 0
        sizes = create_nested_dict(2)
        for set_name, in_dict in grouped.items():
            for instance, in_dict1 in in_dict.items():
                count_of_diff_crossovers = 0
                for local_search, in_dict2 in in_dict1.items():
                    for _, _ in in_dict2.items():
                        count_of_diff_crossovers += 1
                sizes[set_name][instance] = count_of_diff_crossovers
                if set_name == "setD" and instance == "n0000040i07.txt":
                    correct_size = count_of_diff_crossovers
        left_cos_enough_data = create_nested_dict(2)
        for set_name, in_dict in grouped.items():
            for instance, in_dict1 in in_dict.items():
                if sizes[set_name][instance] == correct_size:
                    left_cos_enough_data[set_name][instance] = in_dict1
        return left_cos_enough_data



