from Individual import Individual
import numpy as np

class ValueToReachFoundException(Exception):
	def __init__(self, individual):            
		super().__init__("Value to reach found")
		self.individual = individual


def hamming_distance(individual_a, individual_b):
	return len(np.where(individual_a != individual_b)[0])

