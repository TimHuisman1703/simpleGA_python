import string
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt

from RunStats import RunStats

class StatType(Enum):
    BEST_FITNESS = 1
    AVERAGE_FITNESS = 2

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
    print(values)
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

    plt.show()