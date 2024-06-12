import itertools
import time
import uuid
from pprint import pprint

import numpy as np

import Plotting
from Experiment_writer import ExperimentData, ExperimentSetup
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

unique_batch_id = uuid.uuid4()

def run_configurations_and_save(setUps: [ExperimentSetup]):
    ExperimentData.remember_locally_done_batch(unique_batch_id)
    counter_setup = 0
    for su in setUps:
        counter_setup += 1
        instance_path = f"maxcut-instances/{su.set_name}/{su.instance}"
        options = {}
        if su.selection is not None:
            options['selection'] = su.selection
        if su.mutation is not None:
            options["mutation"] = su.mutation
        if su.offspring is not None:
            options['offspring'] = su.offspring
        if su.local_search is not None:
            options['LocalSearch'] = su.local_search
        if su.crossover is not None:
            options['variation'] = su.crossover
        print(f"Running set-up {counter_setup}/{len(setUps)}")
        print(pprint(vars(su)))
        counter_rerun = 0
        for _ in range(su.times_repeat):
            counter_rerun += 1
            print(f"Running run {counter_rerun}/{su.times_repeat}")
            fitness = FitnessFunction.MaxCut(instance_path)
            if su.crossover == "ECGA":
                genetic_algorithm = ExtendedCompactGeneticAlgorithm(fitness,
                                                                    su.population_size,
                                                                    evaluation_budget=su.max_budget,
                                                                    verbose=False,
                                                                    save_stats=False,
                                                                    **options)
            else:
                genetic_algorithm = GeneticAlgorithm(fitness,
                                                     su.population_size,
                                                     evaluation_budget=su.max_budget,
                                                     verbose=False,
                                                     save_stats=False,
                                                     **options)

            start_time = time.time()
            best_fitness, num_evaluations, has_converged = genetic_algorithm.run()
            time_taken = time.time() - start_time
            if best_fitness == fitness.value_to_reach:
                has_found_opt = True
            else:
                has_found_opt = False
            if "LocalSearch" in su.crossover:
                was_local_search_used = True
            elif su.local_search is not None:
                was_local_search_used = su.local_search
            else:
                was_local_search_used = False
            experiment_data = ExperimentData(su.crossover, was_local_search_used, su.mutation, su.selection, su.offspring,
                                             su.population_size, su.max_budget, su.set_name,
                                             su.number_of_vertices, su.instance, best_fitness,
                                             has_found_opt, num_evaluations, time_taken,
                                             {}, unique_batch_id, has_converged)
            experiment_data.save_run()

def generate_set_ups():
    # Does a cartesian product of all params for now
    instances = get_instances(amount=4, add_mid=True)
    params = {
        "max_budget" : [100000],
        "population_size" : [10, 50, 100, 1000],
        "crossover": ["TwoPointCrossover", "GreedyCrossover", "GreedyMutCrossover", "UniformCrossover"],
        "mutation": [None],
        "selection": [None],
        "offspring": [None],
        "local_search": [False, True],
        "instance": [(50, "setC", [ "n0000050i08.txt"]),
                     (40, "setE", [ "n0000040i09.txt"])],
        "additional_data": [None],
        "times": [10]
    }


    keys = params.keys()
    values = params.values()
    cross_product = itertools.product(*values)

    l_params = [dict(zip(keys, combination)) for combination in cross_product]
    configurations = []

    #TODO: change budget, selection, population, offspring for Qinghua
    for param_set in l_params:
        for (instance_index, instance_name) in enumerate(param_set["instance"][2]):
            # if instance_index != 2: #TODO: Do only your number
            #     continue
            configuration = ExperimentSetup(param_set["crossover"],
                                            param_set["local_search"],
                                            param_set["mutation"],
                                            param_set["selection"],
                                            param_set["offspring"],
                                            param_set["population_size"],
                                            param_set["max_budget"],
                                            param_set["instance"][1],
                                            param_set["instance"][0],
                                            instance_name,
                                            param_set["times"],
                                            param_set["additional_data"]
                                            )
            configurations.append(configuration)


    return configurations

MODE = 'pop'

if __name__ == "__main__":
    # "CustomCrossover", "UniformCrossover", "OnePointCrossover"
    # setups = [
    #     # {'variation': 'TwoPointCrossover'},
    #     {'variation': 'GreedyCrossover'},
    #     {'variation': 'GreedyMutCrossover'},
    #     {'variation': 'UniformCrossover'},
    #     {'variation': 'OnePointCrossover'},
    #     {'variation': 'Qinghua_LocalSearch',
    #      'offspring': 'Qinghua',
    #      'selection': 'BestSolutionsOnly'},
    #     {
    #         'variation': 'GreedyCrossover',
    #         # 'selection': 'FitnessSharing',
    #         # 'offspring': 'SimulatedAnnealing',
    #         'mutation': 'AdaptiveMutation'},
    #     {'variation': 'ECGA'}
    # ]
    # evaluation_dictionary = {}
    # evaluation_budget = 100000
    # population_size = 10

    # configures experiments
    # set_ups = generate_set_ups()
    # run_configurations_and_save(set_ups)


    # data processing
    combined_logs = ExperimentData.load_multiple_runs(["all_runs_0","all_runs_1", "all_runs_2", "all_runs_missing", "all_runs_quing"])
    grouped_data = ExperimentData.group_same_executions(combined_logs)
    averaged_data = ExperimentData.find_average_values_in_grouped(grouped_data)
    # The results are grouped in a nested dictionary indexed by [Set][Instance][Is_local_search][(offspring, selection, mutation, population_size, max_budget)]
    grouped = ExperimentData.group_same_sets_then_instances(averaged_data)
    best_by_population = ExperimentData.leave_best_performing_population(grouped)
    grouped_remove_small_test_size = ExperimentData.remove_small_test_size(best_by_population)

    # version with local search filtered True - only local search, False - no local search
    grouped_filtered_local_search = ExperimentData.filter_local_search(grouped_remove_small_test_size, True)
    sorted_by_name = ExperimentData.sort_by_crossover(grouped_filtered_local_search)

    # version without local search filtering
    # sorted_by_name = ExperimentData.sort_by_crossover(grouped_remove_small_test_size)

    Plotting.plot_performances_on_one_set(sorted_by_name, "setE")






    #print("SDaasd")

    # inst = "maxcut-instances/setE/n0000040i04.txt"
    # for vertex_amount, set_name, instance_names in instances:
    #     print("=" * 100 + "\n")
    #     print(f"Running: {set_name}: {instance_names}, with {vertex_amount} vertices\n")
    #     for instance_name in instance_names:
    #         instance_path = f"maxcut-instances/{set_name}/{instance_name}"
    #         for setup in setups:
    #             variation = setup['variation']
    #             print(f"{variation}")
    #             with open("output-{}.txt".format(setup['variation']),"w") as f:
    #                 num_evaluations_list = []
    #                 num_runs = 10
    #                 num_success = 0
    #                 runs = []
    #                 for i in range(num_runs):
    #                     fitness = FitnessFunction.MaxCut(instance_path)
    #
    #                     if (variation == "ECGA"):
    #                         genetic_algorithm = ExtendedCompactGeneticAlgorithm(fitness,
    #                                                                             population_size,
    #                                                                             evaluation_budget=evaluation_budget,
    #                                                                             verbose=False,
    #                                                                             save_stats=True,
    #                                                                             **setup)
    #                     else:
    #                         genetic_algorithm = GeneticAlgorithm(fitness,
    #                                                              population_size,
    #                                                              evaluation_budget=evaluation_budget,
    #                                                              verbose=False,
    #                                                              save_stats=True,
    #                                                              **setup)
    #
    #                     best_fitness, num_evaluations = genetic_algorithm.run()
    #                     runs.append(genetic_algorithm.statistics)
    #                 plot_runs_per_generation(runs,
    #                                          [f"{i}" for i in range(len(runs))],
    #                                          StatType.BEST_FITNESS,
    #                                          f"Best fitness per generation across runs for {setup['variation']}\nWith {num_success}/{num_runs} reaching the optimum",
    #                                          set_name,
    #                                          f"{vertex_amount}_{instance_name}")
    #
    #                 evaluation_dictionary[setup['variation']] = num_evaluations_list
    #                 print("{}/{} runs successful".format(num_success,num_runs))
    #                 print("{} evaluations (median)".format(np.median(num_evaluations_list)))
    #                 percentiles = np.percentile(num_evaluations_list,[10,50,90])
    #                 f.write("{} {} {} {} {}\n".format(population_size,num_success/num_runs,percentiles[0],percentiles[1],percentiles[2]))
    #                 print()
    #
    #         crossovers = [setup['variation'] for setup in setups]
    #         plot_evaluation_for_crossovers(evaluation_dictionary, crossovers, population_size, evaluation_budget, (vertex_amount, set_name, instance_name))
