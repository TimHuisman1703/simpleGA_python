import numpy as np
from select_instances import get_instances
from Plotting import plot_evaluation_for_crossovers, plot_runs_per_generation, StatType
from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction


if __name__ == "__main__":
    # "CustomCrossover", "UniformCrossover", "OnePointCrossover"
    variations = [ "GreedyCrossoverLocalSearch", "GreedyMutCrossoverLocalSearch"]
    evaluation_dictionary = {}
    evaluation_budget = 100000
    population_size = 1000
    instances = get_instances(amount=1, add_high=True)
    # inst = "maxcut-instances/setE/n0000040i04.txt"
    for vertex_amount, set_name, instance_names in instances:
        print(f"Running: {set_name}: {instance_names}, with {vertex_amount} vertices")
        for instance_name in instance_names:
            instance_path = f"maxcut-instances/{set_name}/{instance_name}"
            for cx in variations:
                with open("output-{}.txt".format(cx),"w") as f:
                    num_evaluations_list = []
                    num_runs = 10
                    num_success = 0
                    runs = []
                    for i in range(num_runs):
                        fitness = FitnessFunction.MaxCut(instance_path)
                        genetic_algorithm = GeneticAlgorithm(fitness, population_size, variation=cx, evaluation_budget=evaluation_budget, verbose=False, save_stats=True)
                        best_fitness, num_evaluations = genetic_algorithm.run()
                        runs.append(genetic_algorithm.statistics)
                        if best_fitness == fitness.value_to_reach:
                            num_success += 1
                        num_evaluations_list.append(num_evaluations)
                    plot_runs_per_generation(runs, [f"{i}" for i in range(len(runs))], StatType.BEST_FITNESS, f"Best fitness per generation across runs for {cx}\nWith {num_success}/{num_runs} reaching the optimum", set_name, f"{vertex_amount}_{instance_name}")

                    evaluation_dictionary[cx] = num_evaluations_list
                    print("{}/{} runs successful".format(num_success,num_runs))
                    print("{} evaluations (median)".format(np.median(num_evaluations_list)))
                    percentiles = np.percentile(num_evaluations_list,[10,50,90])
                    f.write("{} {} {} {} {}\n".format(population_size,num_success/num_runs,percentiles[0],percentiles[1],percentiles[2]))
            plot_evaluation_for_crossovers(evaluation_dictionary, variations, population_size, evaluation_budget, (vertex_amount, set_name, instance_name))
