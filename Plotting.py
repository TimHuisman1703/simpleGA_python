import os
import string
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from RunStats import RunStats

class StatType(Enum):
    BEST_FITNESS = 1
    AVERAGE_FITNESS = 2

def check_next_index_that_exists_file_name(filename: string):
    i = 0
    checked_file = filename
    while os.path.exists(f"graphs/{checked_file}.png"):
        i += 1
        checked_file = f"{filename}_{i}"
    return checked_file

def keep_isalnum_else_underscore(char):
    if char.isalnum():
       return char
    else:
        return "_"

def plot_evaluation_for_crossovers(evaluation_dictionary, crossovers, population_size, evaluations_budget, instance):

    for cx in crossovers:
        values = evaluation_dictionary[cx]
        kde = gaussian_kde(values, bw_method='scott')
        x_grid = np.linspace(0, evaluations_budget, 50)
        kde_values = kde(x_grid)
        kde_values = kde_values / max(kde_values)

        plt.plot(x_grid, kde_values, label=f'Probability Density Function of {cx} evaluations')
        # plt.hist(values, bins=30, density=True, alpha=0.5, color='gray', label='Histogram of values')  # Optional: Show histogram
        plt.scatter(values, np.zeros_like(values), marker='x', label=f'Evaluation data points of {cx}')
    plt.title(f'Probability Distribution of evaluations, population_size={population_size}')
    plt.xlabel('Evaluations')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    instance_name = instance.replace('/', '_').replace('.txt', '').replace('maxcut-instances','')
    if not os.path.exists("graphs/"):
        os.makedirs("graphs")
    plt.savefig(f"graphs/evaluation_plot_crossovers={crossovers}_pop_size={population_size}_budget={evaluations_budget}_instance={instance_name}")
    plt.clf()


# Plots runs and labels them
def plot_runs_per_generation(runs: [RunStats], run_labels: [string], stat_to_plot_name: StatType, plot_name: string):
    values = []
    stat_name = "Initialize me, please!"
    if stat_to_plot_name == StatType.BEST_FITNESS:
        values = [run.best_fitness_per_generation for run in runs]
        stat_name = "Best fitness"
    elif stat_to_plot_name == StatType.AVERAGE_FITNESS:
        values = [run.average_fitness_per_generation for run in runs]
        stat_name = "Average fitness"


    max_length = max(len(arr) for arr in values)
    same_length_values = []
    for arr in values:
        same_length_value = arr + [arr[-1]] * (max_length - len(arr)) # Make sure all lengths match
        same_length_values.append(same_length_value)

    x_values = np.arange(1, max_length + 1)
    for i, arr in enumerate(same_length_values):
        #plt.plot(x_values, arr, label=run_labels[i])
        plt.plot(x_values, arr)

    plt.ylabel(stat_name)
    plt.xlabel("Generations")
    plt.title(plot_name)
    plt.legend()
    unique_name = check_next_index_that_exists_file_name("".join(keep_isalnum_else_underscore(x) for x in plot_name ))
    print(f"name is {unique_name}")
    if not os.path.exists("graphs/"):
        os.makedirs("graphs")
    plt.savefig(f"graphs/{unique_name}")
