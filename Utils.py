from Individual import Individual
import numpy as np

class ValueToReachFoundException(Exception):
	def __init__(self, individual):            
		super().__init__("Value to reach found")
		self.individual = individual


def hamming_distance(individual_a: Individual, individual_b: Individual):
	return len(np.where(individual_a.genotype != individual_b.genotype)[0])

