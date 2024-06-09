import numpy as np

import Plotting
from select_instances import get_instances, get_instance_sets
from Plotting import plot_evaluation_for_crossovers, plot_runs_per_generation, StatType
from GeneticAlgorithm import GeneticAlgorithm
from ExtendedCompactGeneticAlgorithm import ExtendedCompactGeneticAlgorithm
import FitnessFunction



def calculate_avg_eval_to_opt_diff_population(variations, eval_budget, instance_pth, instance, runs_per_pop, min_pop, max_pop, num_pop_thresholds):
    pop_sizes = np.linspace(min_pop, max_pop, num_pop_thresholds, endpoint=True)
    pop_sizes  = list(map(lambda x: round(x) + 1 if round(x) % 2 != 0 else round(x), pop_sizes))
    print(pop_sizes)
    results_per_variation_operator = {}

    dataframe = {
        'Crossover type': [],
        'Evaluation amount': [],
        'Population': [],
        'Evaluations at convergence': [],
    }
    evaluation_keys = ['Population', 'Evaluation amount', 'Crossover type']
    convergence_keys = ['Population', 'Evaluations at convergence', 'Crossover type']

    for cx in variations:
        averages = []
        print(f"Running crossover {cx}")
        for i, pop_size in enumerate(pop_sizes):
            print(f"Running population_size={pop_size}")
            averages.append(0)
            for run in range(runs_per_pop):
                #print(f"Run {run}/{runs_per_pop}")
                fitness = FitnessFunction.MaxCut(instance_pth)
                g_a = GeneticAlgorithm(fitness, pop_size, variation=cx,
                                                     evaluation_budget=eval_budget, verbose=False,  save_stats=False, print_final_results=False)
                best_fitness, num_evaluations, num_converged = g_a.run()
                averages[i] += num_evaluations/runs_per_pop

                # add data to dataframe
                dataframe['Evaluation amount'].append(num_evaluations)
                dataframe['Crossover type'].append(cx)
                dataframe['Population'].append(pop_size)
                dataframe['Evaluations at convergence'].append(num_converged)

        results_per_variation_operator[cx] = averages
    Plotting.plot_average_evaluations_for_different_population_sizes(pop_sizes,results_per_variation_operator, variations, instance, eval_budget)

    dataframe_evaluations = {key: dataframe[key] for key in evaluation_keys}
    dataframe_convergence = {key: dataframe[key] for key in convergence_keys}
    Plotting.boxplot_dataframe(dataframe_evaluations, "regular", instance)
    Plotting.boxplot_dataframe(dataframe_convergence, "convergence", instance)


def calculate_eval_diff_per_crossover_over_set(variations, evaluation_budget, population_size, runs, instances):

    dataframe = {
        'Crossover type': [],
        'Instance': [],
        'Evaluation amount': [],
        'Evaluations at convergence': [],
    }
    evaluation_keys = ['Instance', 'Evaluation amount', 'Crossover type']
    convergence_keys = ['Instance', 'Evaluations at convergence', 'Crossover type']


    for set_name in instances:
        print(f"Running set {set_name}")
        for vertex_amount, instance_names in instances[set_name]:
            for instance_name in instance_names:
                print(f"Running instance {instance_name}, vertices: {vertex_amount}")
                instance_path = f"maxcut-instances/{set_name}/{instance_name}"
                for cx in variations:
                    print(f"Crossover: {cx}")
                    for _ in range(runs):
                        #print(f"Run {run}/{runs_per_pop}")
                        fitness = FitnessFunction.MaxCut(instance_path)
                        g_a = GeneticAlgorithm(fitness, population_size, variation=cx,
                                                             evaluation_budget=evaluation_budget, verbose=False,  save_stats=False, print_final_results=False)
                        _ , num_evaluations, num_converged = g_a.run()

                        # add data to dataframe
                        dataframe['Instance'].append(f"{vertex_amount}_{instance_name}".replace('.txt',''))
                        dataframe['Evaluation amount'].append(num_evaluations)
                        dataframe['Crossover type'].append(cx)
                        dataframe['Evaluations at convergence'].append(num_converged)


        dataframe_evaluations = {key: dataframe[key] for key in evaluation_keys}
        dataframe_convergence = {key: dataframe[key] for key in convergence_keys}
        Plotting.boxplot_dataframe_by_set(dataframe_evaluations, f"regular_pop_{population_size}" ,set_name)
        Plotting.boxplot_dataframe_by_set(dataframe_convergence, f"convergence_pop_{population_size}", set_name)





MODE = 'pop'

if __name__ == "__main__":
    # "CustomCrossover", "UniformCrossover", "OnePointCrossover"
    setups = [
        # {'variation': 'TwoPointCrossover'},
        {'variation': 'GreedyCrossover'},
        {'variation': 'GreedyMutCrossover'},
        {'variation': 'UniformCrossover'},
        {'variation': 'OnePointCrossover'},
        {'variation': 'Qinghua_LocalSearch',
         'offspring': 'Qinghua',
         'selection': 'BestSolutionsOnly'},
        {
            'variation': 'GreedyCrossover',
            # 'selection': 'FitnessSharing',
            # 'offspring': 'SimulatedAnnealing',
            'mutation': 'AdaptiveMutation'},
        {'variation': 'ECGA'}
    ]
    evaluation_dictionary = {}
    evaluation_budget = 100000
    population_size = 10
    instances = get_instances(amount=1)
    # inst = "maxcut-instances/setE/n0000040i04.txt"
    for vertex_amount, set_name, instance_names in instances:
        print("=" * 100 + "\n")
        print(f"Running: {set_name}: {instance_names}, with {vertex_amount} vertices\n")
        for instance_name in instance_names:
            instance_path = f"maxcut-instances/{set_name}/{instance_name}"
            for setup in setups:
                variation = setup['variation']
                print(f"{variation}")
                with open("output-{}.txt".format(setup['variation']),"w") as f:
                    num_evaluations_list = []
                    num_runs = 10
                    num_success = 0
                    runs = []
                    for i in range(num_runs):
                        fitness = FitnessFunction.MaxCut(instance_path)

                        if (variation == "ECGA"):
                            genetic_algorithm = ExtendedCompactGeneticAlgorithm(fitness,
                                                                                population_size,
                                                                                evaluation_budget=evaluation_budget,
                                                                                verbose=False,
                                                                                save_stats=True,
                                                                                **setup)
                        else:
                            genetic_algorithm = GeneticAlgorithm(fitness,
                                                                 population_size,
                                                                 evaluation_budget=evaluation_budget,
                                                                 verbose=False,
                                                                 save_stats=True,
                                                                 **setup)

                        best_fitness, num_evaluations = genetic_algorithm.run()
                        runs.append(genetic_algorithm.statistics)
                        if best_fitness == fitness.value_to_reach:
                            num_success += 1
                        num_evaluations_list.append(num_evaluations)
                    plot_runs_per_generation(runs,
                                             [f"{i}" for i in range(len(runs))],
                                             StatType.BEST_FITNESS,
                                             f"Best fitness per generation across runs for {setup['variation']}\nWith {num_success}/{num_runs} reaching the optimum",
                                             set_name,
                                             f"{vertex_amount}_{instance_name}")

                    evaluation_dictionary[setup['variation']] = num_evaluations_list
                    print("{}/{} runs successful".format(num_success,num_runs))
                    print("{} evaluations (median)".format(np.median(num_evaluations_list)))
                    percentiles = np.percentile(num_evaluations_list,[10,50,90])
                    f.write("{} {} {} {} {}\n".format(population_size,num_success/num_runs,percentiles[0],percentiles[1],percentiles[2]))
                    print()

            crossovers = [setup['variation'] for setup in setups]
            plot_evaluation_for_crossovers(evaluation_dictionary, crossovers, population_size, evaluation_budget, (vertex_amount, set_name, instance_name))
