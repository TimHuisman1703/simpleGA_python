import numpy as np

from Plotting import plot_runs_per_generation, StatType
from GeneticAlgorithm import GeneticAlgorithm
import FitnessFunction

if __name__ == "__main__":
	crossovers = [ "UniformCrossover"]
	for cx in crossovers:
		inst = "maxcut-instances/setE/n0000040i04.txt"
		with open("output-{}.txt".format(cx),"w") as f:
			population_size = 500
			num_evaluations_list = []
			num_runs = 30
			num_success = 0
			runs = []
			for i in range(num_runs):
				fitness = FitnessFunction.MaxCut(inst)
				genetic_algorithm = GeneticAlgorithm(fitness,population_size,variation=cx,evaluation_budget=100000,verbose=False, save_stats=True)

				best_fitness, num_evaluations = genetic_algorithm.run()
				runs.append(genetic_algorithm.statistics)
				if best_fitness == fitness.value_to_reach:
					num_success += 1
				num_evaluations_list.append(num_evaluations)
			plot_runs_per_generation(runs, [f"{i}" for i in range(len(runs))], StatType.BEST_FITNESS, f"Best fitness per generation across runs for {cx}\nWith {num_success}/{num_runs} reaching the optimum")
			print("{}/{} runs successful".format(num_success,num_runs))
			print("{} evaluations (median)".format(np.median(num_evaluations_list)))
			percentiles = np.percentile(num_evaluations_list,[10,50,90])
			f.write("{} {} {} {} {}\n".format(population_size,num_success/num_runs,percentiles[0],percentiles[1],percentiles[2]))