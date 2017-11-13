#Start
#Evaluate Fitngas
#New Population
	#Selection - roulette
	#Crossover
	#mutation
	#Accepting - add to new pop
#Replace - old pop with the new
#Test
#Loop
import sys
import operator
from math import exp, sqrt
import random
import neural

class GA:
	print("something")	#remove this
	#class vars
	num_input_nodga = 2
	num_output_nodga = 1
	num_hidden_layers = 2
	num_hidden_nodga = 10
	population_size = 20          #parent generation size >>>TUNABLE<<<
	num_offspring = 40            #child generation size >>>TUNABL<<<
	gen_till_convergence = 5     #number of generations with no change before "convergence" is determined. >>>TUNABLE<<<
	weight_upper_bound = sys.maxint = 1000000
	weight_lower_bound = -sys.maxint
	
	#data specific
	num_inputs = 5
	num_outputs = 1
	
	population = [None] * population_size
	offspring = [None] * num_offspring
	generation = 0
	gen_since_change = 0
	best_individual = None
	best_individual_score = None
	
	def __init__(self):
		print("init")
		#create random chromosomga
		for x in range(0, self.population_size):
			sigma = random.uniform(0, 5)
			in_hidden_edgga = self.num_input_nodga * self.num_hidden_nodga
			hidden_hidden_edgga = 0
			for i in range(self.num_hidden_layers -1):
				hidden_hidden_edgga += self.num_hidden_nodga * self.num_hidden_nodga
			hidden_out_edgga = self.num_hidden_nodga * self.num_output_nodga
			individual = [None] * (in_hidden_edgga + hidden_hidden_edgga + hidden_out_edgga)
			for y in range(0, len(individual)):
				individual[y] = random.uniform(-1, 1)
			self.population[x] = [individual, sigma]
			
		#initializers
		
	def crossover(self, p1, p2):
		
		return p1
	
	
	def mutate(self, individual):
		weights = individual[0]
		sigma = individual[1]
		u = random.uniform(0,1)
		sigma = sigma * exp(u/sqrt(len(weights)))       #mutate the sigma value
		for index, wx in enumerate(weights):            #mutate the weights
			shift = random.uniform(0, int(sigma))       #becaue sigma is real, we need to cast it to keep the weights as integers.
			wx = wx + shift
			wx = max(wx, self.weight_lower_bound)
			wx = min(wx, self.weight_upper_bound)
			weights[index] = wx
		return [weights, sigma]
		
	def evaluate(self, individual):
		weights = individual[0]
		point1 = self.num_input_nodga*self.num_hidden_nodga
		point2 = self.num_input_nodga*self.num_hidden_nodga+(self.num_hidden_nodga*(self.num_hidden_layers-1)*self.num_hidden_nodga)
		input = weights[:point1]
		hidden = weights[point1:point2]
		output = weights[point2:len(weights)]
		nn = neural.NeuralNetwork()
		error = nn.forward_propagate(input,hidden,output)
		return error
	
	
if __name__ == '__main__':
	ga = GA()
	score1 = ga.evaluate(ga.population[5])
	print(score1)
	while(ga.gen_since_change < ga.gen_till_convergence):
		ga.generation = ga.generation + 1                       #increase generation count
		ga.gen_since_change = ga.gen_since_change + 1           #increase non-changing generation count
		#roulette selection
		for i in range(0, ga.num_offspring):                    #generate offspring
			chromo1 = ga.population[random.randint(0, ga.population_size - 1)]  #add probablilities based on fitness
			chromo2 = ga.population[random.randint(0, ga.population_size - 1)]
			#do crossover
			os = ga.crossover(chromo1, chromo2)
			ga.offspring[i] = ga.mutate(os) #mutate
		all =  ga.population + ga.offspring
		scores = list()
		for i in range(0, len(all)):
			scores.append([all[i], ga.evaluate(all[i])])
			if(ga.best_individual_score is None or scores[i][1] < ga.best_individual_score ):         #if there is a new best chromosome
				print("Updated best individual.")
				ga.best_individual = all[i]
				ga.best_individual_score = scores[i][1]
				ga.gen_since_change = 0
		scores.sort()
		#add best individuals to next population
		for i in range(0, ga.population_size):
			ga.population[i] = scores[i][0]
		
		
	print(ga.best_individual_score)
		
		
		
		
		