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
    plt.clf()

    height = 0
    # set figure size so legend not too big
    for cx in crossovers:
        values = evaluation_dictionary[cx]
        plt.scatter(values, np.repeat(height, len(values)), marker='x', label=f'Data points of {cx}')
        height += 0.05
        # if all avlues the same, dont plot distribution, may crash
        if values.count(values[0]) == len(values):
            continue
        kde = gaussian_kde(values, bw_method='scott')
        x_grid = np.linspace(0, max(values), 50)
        kde_values = kde(x_grid)
        kde_values = kde_values / max(kde_values)

        plt.plot(x_grid, kde_values, label=f'PDF of {cx} evaluations')

    plt.rcParams["figure.figsize"] = (28, 22)
    plt.title(f'Probability Distribution of evaluations, population_size={population_size}')
    plt.xlabel('Evaluations')
    plt.ylabel('Density')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)


    # save image
    instance_name = f"{instance[1]}_{instance[2]}_vertices={instance[0]}".replace('.txt','')
    if not os.path.exists("graphs/"):
        os.makedirs("graphs")
    plt.savefig(f"graphs/evaluation_plot_crossovers={crossovers}_pop_size={population_size}_budget={evaluations_budget}_instance={instance_name}", bbox_inches='tight')

    # plt.show()


# Plots runs and labels them
def plot_runs_per_generation(runs: [RunStats], run_labels: [string], stat_to_plot_name: StatType, plot_name: string, set_name: string, instance_name: string):
    plt.clf()
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
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    unique_name = check_next_index_that_exists_file_name("".join(keep_isalnum_else_underscore(x) for x in plot_name ))
    print(f"name is {unique_name}")
    if not os.path.exists(f"graphs/{set_name}/{instance_name}"):
        os.makedirs(f"graphs/{set_name}/{instance_name}")
    plt.savefig(f"graphs/{set_name}/{instance_name}/{unique_name}")

def plot_average_evaluations_for_different_population_sizes(population_sizes, average_evaluations_dict, crossovers, instance, evaluations_budget):
    plt.clf()
    counter = -1
    for cx in crossovers:
        counter += 1
        print(cx)
        values = average_evaluations_dict[cx]
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k'][counter % 7]
        plt.scatter(population_sizes, values, color=color, label=f"Averages for {cx}")

        # Calculate the best fit line
        coefficients = np.polyfit(population_sizes, values, 2)  # 1st degree polynomial (linear fit)
        poly = np.poly1d(coefficients)
        y_fit = poly(population_sizes)

        # Plot the best fit line
        plt.plot(population_sizes, y_fit, "--", color=color, label=f"Best fit line for {cx}")

    plt.title("Average number of evaluations to reach optimum")
    plt.xlabel('Population size')
    plt.ylabel('Average number of evaluations')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    if not os.path.exists(f"graphs/evaluations_based_on_pop/{instance[1]}/{instance[2]}"):
        os.makedirs(f"graphs/evaluations_based_on_pop/{instance[1]}/{instance[2]}")
    instance_name = f"{instance[1]}_{instance[2]}_vertices={instance[0]}".replace('.txt', '')
    plt.savefig(
        f"graphs/evaluations_based_on_pop/{instance[1]}/{instance[2]}/budget={evaluations_budget}_instance={instance_name}",
        bbox_inches='tight')

