import sys
from csv import reader
import numpy as np

#configuration parameters
num_input_nodes = 2
num_output_nodes = 1
num_hidden_layers = 2
num_hidden_nodes = 10         #per-layer

#load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file, delimiter='\t')
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
    
#change data string to number
def str_to_float(dataset):
    for i in range(len(dataset[0])):
        for row in dataset:
            row[i] = float(row[i].strip())

#Linearly normalize the whole data set by min max column values.
def normalize_data(dataset):
    datamin = [min(column) for column in zip(*dataset)]
    datamax = [max(column) for column in zip(*dataset)]
    for row in dataset:
        for i in range(len(row)):
            row[i] = (float(row[i]) - datamin[i]) / (datamax[i] - datamin[i])
            
    
def evaluate_network(network):
    return
    
def forward_propagate(network, dataset):
    sum_error = 0
    for i in range(0, data_set_size):
        test_index = random.randint(0, data_set_size)
        data = np.array(dataset[test_index])
        input_to_hidden = np.cross(data, network[0])
        input_to_hidden = np.tanh(input_to_hidden)
        hidden_to_hidden = np.cross(input_to_hidden, network[1])
        hidden_to_hidden = np.tanh(hidden_to_hidden)
        hidden_to_out = np.sum(np.cross(hidden_to_hidden, network[2]))
    return sum_error

def create_network(list_i_h_edges, list_h_h_edges, list_h_o_edges):
    num_weights = num_input_nodes * num_hidden_nodes + num_hidden_nodes * num_hidden_nodes + num_hidden_nodes * num_output_nodes
#    print("Num_weights: "+str(num_weights))
    network = [[None]*num_weights for i in range(num_weights)]
#    print("Len Network: "+str(len(network)))
#    print("Len Network[1]: "+str(len(network[1])))
#    print("Len of ih: "+str(len(list_i_h_edges))+" len of hh: "+str(len(list_h_h_edges))+" len of ho: "+str(len(list_h_o_edges)))
    for i in range(num_input_nodes-1):
        for j in range(num_input_nodes, num_input_nodes + num_hidden_nodes-1):
            network[i][j] = list_i_h_edges[i]
    for i in range(num_input_nodes, num_input_nodes + num_hidden_nodes-1):
        for j in range(num_input_nodes + num_hidden_nodes, num_input_nodes + num_hidden_nodes + num_hidden_nodes-1):
#           print("Network["+str(i)+"]["+str(j)+"]: "+str(len(network[i])))
            network[i][j] = list_h_h_edges[i-(num_input_nodes + num_hidden_nodes)]
    for i in range(num_input_nodes + num_hidden_nodes, num_input_nodes + num_hidden_nodes + num_hidden_nodes - 1):
        for j in range(num_input_nodes + num_hidden_nodes + num_hidden_nodes, num_input_nodes + num_hidden_nodes + num_hidden_nodes + num_output_nodes-1):
            network[i][j] = list_h_o_edges[i-(num_input_nodes + num_hidden_nodes + num_hidden_nodes)]
    network = np.array(network)
    return network
    
    
#RUN
dataset = load_csv('2dData.tsv')
str_to_float(dataset)
#print(dataset)
normalize_data(dataset)
#print(dataset)