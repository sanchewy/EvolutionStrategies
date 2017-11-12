import sys
import operator
from math import exp, expm1, sqrt
import random
import neural

class EvolutionStrategy:
    #configuration parameters
    num_input_nodes = 11
    num_output_nodes = 1
    num_hidden_layers = 2
    num_hidden_nodes = 10         #per-layer
    #remember that the number_offspring is usually considerably higher than the parent generation size (populatino_size)
    population_size = 20          #parent generation size >>>TUNABLE<<<
    num_offspring = 40            #child generation size >>>TUNABL<<<
    gen_till_convergence = 4     #number of generations with no change before "convergence" is determined. >>>TUNABLE<<<
    init_sigma_bounds = 50         #range in which the initial sigma variance values are set.
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
    best_individual_score = None   #best individual's score (evaluation method return)

    def __init__(self):
        init_bounds = sqrt(6/(self.num_inputs+self.num_outputs))
        for x in range(0, self.population_size):
            sigma = random.uniform(0, self.init_sigma_bounds)
            in_hidden_edges = self.num_input_nodes * self.num_hidden_nodes
            hidden_hidden_edges = 0
            for i in range(self.num_hidden_layers -1):
                hidden_hidden_edges += self.num_hidden_nodes * self.num_hidden_nodes
            hidden_out_edges = self.num_hidden_nodes * self.num_output_nodes
            individual = [None] * (in_hidden_edges + hidden_hidden_edges + hidden_out_edges)
            for y in range(0, len(individual)):
                individual[y] = random.uniform(-init_bounds, init_bounds)
            self.population[x] = [individual, sigma]

    #self-adaptive guassian mutation with weights on per-chromosome basis rather than element-index specific
    #this method mutates a single chromosome
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

    #create a neural net corresponding to individual and evalute its performance.
    def evaluate(self, individual,databreak):
        weights = individual[0]
        point1 = self.num_input_nodes*self.num_hidden_nodes
        point2 = self.num_input_nodes*self.num_hidden_nodes+(self.num_hidden_nodes*(self.num_hidden_layers-1)*self.num_hidden_nodes)
        input = weights[:point1]
        hidden = weights[point1:point2]
        output = weights[point2:len(weights)]
        nn = neural.NeuralNetwork(databreak)
        # net = nn.create_network(input,hidden,output)
        error = nn.forward_propagate(input,hidden,output)
        return error

    def final_eval(self, individual, databreak):
        weights = individual[0]
        point1 = self.num_input_nodes*self.num_hidden_nodes
        point2 = self.num_input_nodes*self.num_hidden_nodes+(self.num_hidden_nodes*(self.num_hidden_layers-1)*self.num_hidden_nodes)
        input = weights[:point1]
        hidden = weights[point1:point2]
        output = weights[point2:len(weights)]
        nn = neural.NeuralNetwork(databreak)
        # net = nn.create_network(input,hidden,output)
        error = nn.final_eval(input,hidden,output)
        return error

if __name__ == '__main__':
    es = EvolutionStrategy()
    best_individuals = list()
    final_errors = list()
    num_generations = list()
    for databreak in range(5):
        while(es.gen_since_change < es.gen_till_convergence):       #while we havent converged
            print("Generations since last change to best individual: "+str(es.gen_since_change))
            es.generation = es.generation + 1                       #increase generation count
            es.gen_since_change = es.gen_since_change + 1           #increase non-changing generation count
            for i in range(0, es.num_offspring):                    #generate offspring
                chromo = es.population[random.randint(0, es.population_size - 1)]
                es.offspring[i] = es.mutate(chromo)
            merged_pop = es.population + es.offspring
            scores = list()
            for i in range(0, len(merged_pop)):                      #evaluate chromosomes
                scores.append([merged_pop[i], es.evaluate(merged_pop[i],databreak)])
                if(es.best_individual_score is None or scores[i][1] < es.best_individual_score ):         #if there is a new best chromosome
                    print("Updated best individual.")
                    es.best_individual = merged_pop[i]
                    es.best_individual_score = scores[i][1]
                    es.gen_since_change = 0                          #reset non=changing generation count
            scores.sort(key=lambda x:x[1])
            for i in range(0, es.population_size):                   #put the |population_size| best individuals into next generation
                # print("scores[i][1]: "+str(scores[i][1]))
                print("Population updated.")
                es.population[i] = scores[i][0]
                #print("Scores: "+str(scores[i]))
            # print()
        print("Crosss validation fold "+str(databreak)+" finished.")
        best_individuals.append(es.best_individual)
        final_errors.append(es.final_eval(es.best_individual, databreak))
        num_generations.append(es.generation)
        es.gen_since_change = 0
        es.best_individual, es.best_individual_score, es.generation = None, None, 0
    print("Best individual weight matrix found. \nIndividual: %s.\n Scores: %s.\n Generations to train: %s." %(str(best_individuals[0]), str(final_errors), str(num_generations)))
