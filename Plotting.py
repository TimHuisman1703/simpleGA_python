import os
import string
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.stats import gaussian_kde
from scipy.stats import ttest_ind_from_stats

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


def boxplot_dataframe(dataframe, name, instance):
    plt.clf()
    plt.figure(figsize=(12,6))
    # set figure size so legend not too big
    df = pd.DataFrame(dataframe)
    x_name = df.columns[0]
    y_name = df.columns[1]
    hue_name = df.columns[2]

    sns.boxplot(x=x_name, y=y_name, hue=hue_name, data=df, gap=.1, medianprops={"color": "r", "linewidth": 2},)

    plt.title(f'Evaluations per crossover by population sizes, instance: {instance}')
    plt.xlabel(x_name)
    plt.xticks(rotation=90,fontsize=9)
    plt.ylabel(y_name)
    plt.yticks(fontsize=9)


    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    graph_name = f'{name}_evaluations_per_crossover_by_population_{instance[0]}_{instance[1]}_{instance[2]}'

    if not os.path.exists(f"graphs/evaluations_based_on_pop/{instance[1]}/{instance[2]}"):
        os.makedirs(f"graphs/evaluations_based_on_pop/{instance[1]}/{instance[2]}")

    plt.savefig(f"graphs/evaluations_based_on_pop/{instance[1]}/{instance[2]}/{graph_name}", bbox_inches='tight')


def boxplot_dataframe_by_set(dataframe, name, set_name):
    plt.clf()
    plt.figure(figsize=(12,6))
    # set figure size so legend not too big
    df = pd.DataFrame(dataframe)
    x_name = df.columns[0]
    y_name = df.columns[1]
    hue_name = df.columns[2]

    sns.boxplot(x=x_name, y=y_name, hue=hue_name, data=df, gap=.1, medianprops={"color": "r", "linewidth": 2},)

    plt.title(f'Evaluations per crossover on set: {set_name}')
    plt.xlabel(x_name)
    plt.xticks(rotation=90,fontsize=9)
    plt.ylabel(y_name)
    plt.yticks(fontsize=9)


    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    graph_name = f'{name}_evaluations_per_crossover_on_{set_name}'

    if not os.path.exists(f"graphs/set_evaluations/{set_name}"):
        os.makedirs(f"graphs/set_evaluations/{set_name}")

    plt.savefig(f"graphs/set_evaluations/{set_name}/{graph_name}", bbox_inches='tight')



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
    name = f"graphs/_pop_size={population_size}_budget={evaluations_budget}_instance={instance_name}"
    plt.savefig(name, bbox_inches='tight')

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

        coefficients = np.polyfit(population_sizes, values, 2)
        poly = np.poly1d(coefficients)
        y_fit = poly(population_sizes)

        plt.plot(population_sizes, y_fit, "--", color=color, label=f"Best fit for {cx}")

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

def plot_performances_on_one_set(sorted_by_name_performances_same_set, setname, fig=None, ax=None, index=0, last_set=False):
    create_plots = fig is None or ax is None
    sorted_by_name_performances_same_set = sorted_by_name_performances_same_set[setname]
    instances = []
    performances_by_group = {}
    names = []
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams.update({'font.size': 14})
    for instance, list_of_setups in sorted_by_name_performances_same_set.items():

        if len(names) > 0:
            break
        names = []
        for setup in list_of_setups:
            if setup[0][1]:
                names.append(f"{setup[0][0]} with LS")
            else:
                names.append(f"{setup[0][0]}")
    colors = colors[:12] = colors[:12]

    for instance, list_of_setups in sorted_by_name_performances_same_set.items():
        instances.append(instance)
        performances_by_group[instance] = list(map(lambda x: (x[1]["avg_budget_used"], x[1]["variance"]), list_of_setups))

    if create_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = [ax]

    # Define the x positions for each group
    x_positions = np.arange(len(instances))

    group_width = 0.8
    strip_width = group_width / 8  # Divide the group space into 8 strips

    table = f'\\multirow{{{len(instances)}}}{{*}}{{{setname[-1]}}} '
    # Plot each group
    for i, group in enumerate(instances):
        averages, variances = zip(*performances_by_group[group])
        y_positions = np.array(averages)
        errors = (np.power(np.array(variances), 0.5))
        best_avg, best_var = min(zip(averages, errors), key=lambda x: x[0])

        table += f'& {group} & '
        for j, (avg, err) in enumerate(zip(averages, errors)):
            if y_positions[j] == min(y_positions):
                table += f'\\textbf{{{avg:.1f}}} $\\pm$ \\textbf{{{err:.1f}}}$^{{\\dagger,*}}$ & '
            else:
                # perform ttest with standard deviation correction
                stats = ttest_ind_from_stats(avg, err * np.sqrt(10/9), 10, best_avg, best_var * np.sqrt(10/9), 10, equal_var=False)
                if stats.pvalue >= 0.05 and err < avg / 2:
                    # cannot reject the null hypothesis, the means can be equal
                    table += f'{avg:.1f} $\\pm$ {err:.1f}$^{{\\dagger,*}}$& '
                elif stats.pvalue >= 0.05:
                    table += f'{avg:.1f} $\\pm$ {err:.1f}$^\\dagger$ & '
                else:
                    table += f'{avg:.1f} $\\pm$ {err:.1f} & '
        table = table[:-2] + '\\\\\n'

        group_x_positions = x_positions[i] + np.linspace(-group_width / 2 + strip_width / 2,
                                                         group_width / 2 - strip_width / 2, len(y_positions))

        for j, (x, y, err) in enumerate(zip(group_x_positions, y_positions, errors)):
            ax[index].errorbar(x, y, yerr=err, fmt='o', color=colors[j], capsize=5,
                        label=f'{names[j]}' if i == 0 else "")

    print(table, end='\\hline\n')

    if create_plots:
        # Customize the plot
        ax[index].set_xticks(x_positions)
        ax[index].set_xticklabels(instances)
        ax[index].set_xlabel('Instances')
        ax[index].set_ylabel('Evaluations')
        ax[index].legend(loc='upper left', framealpha=0.5)
        ax[index].set_title(f'Average evaluations to reach optimum for {setname}')
        plt.tight_layout()
        plt.savefig(f'graphs/results/{setname}_evaluations.png')
        plt.show(bbox_inches='tight')
    else:
        # Customize the plot
        ax[index].set_xticks(x_positions)
        ax[index].set_xticklabels(instances)
        ax[index].set_xlabel('Instances')
        ax[index].set_ylabel('Evaluations')
        ax[index].set_title(f'Average evaluations to reach optimum for {setname}')
        ax[index].legend(loc='upper left', framealpha=0.5)

def plot_performances_on_all_sets(sorted_by_name_performances_same_set):
    # fig, ax = plt.subplots(1, 5, figsize=(6 * 5, 6))
    for index, setname in enumerate(sorted(sorted_by_name_performances_same_set.keys())):
        plot_performances_on_one_set(sorted_by_name_performances_same_set,
                                     setname,
                                     None,
                                     None,
                                     0,
                                     index == len(sorted_by_name_performances_same_set.keys()) - 1)
    # plt.tight_layout()
    # plt.show()
