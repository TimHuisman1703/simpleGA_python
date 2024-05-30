import numpy as np

import Plotting
from select_instances import get_instances
from Plotting import plot_evaluation_for_crossovers, plot_runs_per_generation, StatType
from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

def calculate_avg_eval_to_opt_diff_population(variations, eval_budget, instance_pth, instance, runs_per_pop, min_pop, max_pop, num_pop_thresholds):
    pop_sizes = np.linspace(min_pop, max_pop, num_pop_thresholds, endpoint=True)
    pop_sizes  = list(map(lambda x: round(x) + 1 if round(x) % 2 != 0 else round(x), pop_sizes))
    print(pop_sizes)
    results_per_variation_operator = {}

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
                best_fitness, num_evaluations = g_a.run()
                averages[i] += num_evaluations/runs_per_pop
        results_per_variation_operator[cx] = averages
    Plotting.plot_average_evaluations_for_different_population_sizes(pop_sizes,results_per_variation_operator, variations, instance, eval_budget)


if __name__ == "__main__":
    # "CustomCrossover", "UniformCrossover", "OnePointCrossover"
    variations = [ "UniformCrossoverLocalSearch", "GreedyMutCrossover", "GreedyCrossover",  "GreedyCrossoverLocalSearch", "GreedyMutCrossoverLocalSearch"]
    evaluation_dictionary = {}
    evaluation_budget = 100000
    population_size = 1000
    instances = get_instances(amount=1, add_low=True, add_mid=True)
    # inst = "maxcut-instances/setE/n0000040i04.txt"
    for vertex_amount, set_name, instance_names in instances:
        print(f"Running: {set_name}: {instance_names}, with {vertex_amount} vertices")
        for instance_name in instance_names:
            instance_path = f"maxcut-instances/{set_name}/{instance_name}"
            print(instance_name)
            if set_name != "setA" and set_name != "setB":
                calculate_avg_eval_to_opt_diff_population(variations, evaluation_budget, instance_path, (vertex_amount, set_name, instance_name.replace('.txt', '')), 5, 30, 1000, 10)
            #for cx in variations:
            #
            #    with open("output-{}.txt".format(cx),"w") as f:
            #         num_evaluations_list = []
            #         num_runs = 10
            #         num_success = 0
            #         runs = []
            #         for i in range(num_runs):
            #             fitness = FitnessFunction.MaxCut(instance_path)
            #             genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=evaluation_budget, verbose=False, save_stats=True)
            #             best_fitness, num_evaluations = genetic_algorithm.run()
            #             runs.append(genetic_algorithm.statistics)
            #             if best_fitness == fitness.value_to_reach:
            #                 num_success += 1
            #             num_evaluations_list.append(num_evaluations)
            #         plot_runs_per_generation(runs, [f"{i}" for i in range(len(runs))], StatType.BEST_FITNESS, f"Best fitness per generation across runs for {cx}\nWith {num_success}/{num_runs} reaching the optimum", set_name, f"{vertex_amount}_{instance_name}")
            #
            #         evaluation_dictionary[cx] = num_evaluations_list
            #         print("{}/{} runs successful".format(num_success,num_runs))
            #         print("{} evaluations (median)".format(np.median(num_evaluations_list)))
            #         percentiles = np.percentile(num_evaluations_list,[10,50,90])
            #         f.write("{} {} {} {} {}\n".format(population_size,num_success/num_runs,percentiles[0],percentiles[1],percentiles[2]))
            # plot_evaluation_for_crossovers(evaluation_dictionary, variations, population_size, evaluation_budget, (vertex_amount, set_name, instance_name))
