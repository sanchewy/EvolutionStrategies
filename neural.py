import sys
from csv import reader
import numpy as np
import random

class NeuralNetwork:
    #configuration parameters
    num_input_nodes = 2
    num_output_nodes = 1
    num_hidden_layers = 2
    num_hidden_nodes = 10         #per-layer
    data_set_location = "2dData.tsv"
    dataset = list()
    
    def __init__(self):
        dataset = self.load_csv(self.data_set_location)
        #print(dataset)
        self.str_to_float(dataset)
        self.normalize_data(dataset)
        self.dataset = dataset
        
    #load a CSV file
    def load_csv(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file, delimiter='\t')
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset
        
    #change data string to number
    def str_to_float(self, dataset):
        for i in range(len(dataset[0])):
            for row in dataset:
                row[i] = float(row[i].strip())

    #Linearly normalize the whole data set by min max column values.
    def normalize_data(self, dataset):
        datamin = [min(column) for column in zip(*self.dataset)]
        datamax = [max(column) for column in zip(*self.dataset)]
        for row in self.dataset:
            for i in range(len(row)):
                row[i] = (float(row[i]) - datamin[i]) / (datamax[i] - datamin[i])
                
        
    def evaluate_network(self, network):
        return
        
    def forward_propagate(self, network):
        sum_error = 0
        for i in range(0, len(self.dataset)):
            test_index = random.randint(0, len(self.dataset))
            data = np.array(self.dataset[test_index])
            input_to_hidden = np.dot(data, network[0])
            input_to_hidden = np.tanh(input_to_hidden)
            hidden_to_hidden = np.dot(input_to_hidden, network[1])
            hidden_to_hidden = np.tanh(hidden_to_hidden)
            hidden_to_out = np.sum(np.dot(hidden_to_hidden, network[2]))
        return sum_error

    def create_network(self, list_i_h_edges, list_h_h_edges, list_h_o_edges):
        num_weights = self.num_input_nodes * self.num_hidden_nodes + self.num_hidden_nodes * self.num_hidden_nodes + self.num_hidden_nodes * self.num_output_nodes
    #    print("Num_weights: "+str(num_weights))
        network = [[None]*num_weights for i in range(num_weights)]
    #    print("Len Network: "+str(len(network)))
    #    print("Len Network[1]: "+str(len(network[1])))
    #    print("Len of ih: "+str(len(list_i_h_edges))+" len of hh: "+str(len(list_h_h_edges))+" len of ho: "+str(len(list_h_o_edges)))
        for i in range(self.num_input_nodes-1):
            for j in range(self.num_input_nodes, self.num_input_nodes + self.num_hidden_nodes-1):
                network[i][j] = list_i_h_edges[i]
        for i in range(self.num_input_nodes, self.num_input_nodes + self.num_hidden_nodes-1):
            for j in range(self.num_input_nodes + self.num_hidden_nodes, self.num_input_nodes + self.num_hidden_nodes + self.num_hidden_nodes-1):
    #           print("Network["+str(i)+"]["+str(j)+"]: "+str(len(network[i])))
                network[i][j] = list_h_h_edges[i-(self.num_input_nodes + self.num_hidden_nodes)]
        for i in range(self.num_input_nodes + self.num_hidden_nodes, self.num_input_nodes + self.num_hidden_nodes + self.num_hidden_nodes - 1):
            for j in range(self.num_input_nodes + self.num_hidden_nodes + self.num_hidden_nodes, self.num_input_nodes + self.num_hidden_nodes + self.num_hidden_nodes + self.num_output_nodes-1):
                network[i][j] = list_h_o_edges[i-(self.num_input_nodes + self.num_hidden_nodes + self.num_hidden_nodes)]
        network = np.array(network)
        return network