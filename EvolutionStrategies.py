import sys
import operator
from math import exp, expm1, sqrt
import random

class EvolutionStrategy:
    #configuration parameters
    num_hidden_layers = 2
    num_hidden_nodes = 10         #per-layer
    #remember that the number_offspring is usually considerably higher than the parent generation size (populatino_size)
    population_size = 20          #parent generation size >>>TUNABLE<<<
    num_offspring = 40            #child generation size >>>TUNABL<<<
    gen_till_convergence = 25     #number of generations with no change before "convergence" is determined. >>>TUNABLE<<<
    init_sigma_bounds = 5         #range in which the initial sigma variance values are set. 
    weight_upper_bound = sys.maxint = 1000000
    weight_lower_bound = -sys.maxint 
    
    #these depend on the data set being used. They are only used for chromosome initial bounds.
    num_inputs = 5
    num_outputs = 1
    
    #class variables
    population = [None] * population_size
    offspring = [None] * num_offspring
    generation = 0
    gen_since_change = 0        #for convergence, count the number of generations since an improvement was made
    best_individual = None      #keep track of the best performing individual so far
    best_individual_score = -sys.maxint   #best individual's score (evaluation method return)      
    
    def __init__(self):
        init_bounds = sqrt(6/(self.num_inputs+self.num_outputs))
        for x in range(0, self.population_size):
            sigma = random.uniform(0, self.init_sigma_bounds)
            individual = [None] * (self.num_hidden_layers*self.num_hidden_nodes)
            for y in range(0, self.num_hidden_layers*self.num_hidden_nodes):
                individual[y] = random.randint(-init_bounds, init_bounds)
            self.population[x] = [individual, sigma]
    
    #self-adaptive guassian mutation with weights on per-chromosome basis rather than element-index specific
    #this method mutates a single chromosome
    def mutate(self, individual):
        weights = individual[0]
        sigma = individual[1]
        u = random.uniform(0,1)
        sigma = sigma * exp(u/sqrt(len(weights)))       #mutate the sigma value
        for index, wx in enumerate(weights):            #mutate the weights
            shift = random.randint(0, int(sigma))       #becaue sigma is real, we need to cast it to keep the weights as integers.
            wx = wx + shift
            wx = max(wx, self.weight_lower_bound)
            wx = min(wx, self.weight_upper_bound)
            weights[index] = wx
        return [weights, sigma]
   
    #create a neural net corresponding to individual and evalute its performance.
    def evaluate(self, individual):
        weights = individual[0]
        #TODO: Call to neural network with weight vector and return the error metric.
        return 0
        
if __name__ == '__main__':
    es = EvolutionStrategy()
    while(es.gen_since_change < es.gen_till_convergence):       #while we havent converged
        es.generation = es.generation + 1                       #increase generation count  
        es.gen_since_change = es.gen_since_change + 1           #increase non-changing generation count
        for i in range(0, es.num_offspring):                    #generate offspring
            chromo = es.population[random.randint(0, es.population_size - 1)]
            es.offspring[i] = es.mutate(chromo)
        merged_pop = es.population + es.offspring
        scores = [None] * len(merged_pop)
        for i in range(0, len(merged_pop)):                      #evaluate chromosomes
            scores[i] = [i, es.evaluate(merged_pop[i])]
            if(scores[i][1] > es.best_individual_score):         #if there is a new best chromosome
                es.best_individual = merged_pop[i]
                es.best_individual_score = scores[i][1]
                es.gen_since_change = 0                          #reset non=changing generation count
        for i in range(0, es.population_size):                   #put the |population_size| best individuals into next generation
            index, value = max(scores,key=operator.itemgetter(0))
            es.population[i] = merged_pop[index]
            scores.remove([index, value])
    print("Best individual weight matrix found. Individual: %s Score: %d." %(str(es.best_individual), es.best_individual_score))