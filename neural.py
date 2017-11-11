import sys
from csv import reader

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
        print(i)
        for row in dataset:
            row[i] = float(row[i].strip())

#Linearly normalize the whole data set by min max column values.
def normalize_data(dataset):
    datamin = [min(column) for column in zip(*dataset)]
    print(str(datamin))
    datamax = [max(column) for column in zip(*dataset)]
    print(str(datamax))
    for row in dataset:
        for i in range(len(row)):
            row[i] = (float(row[i]) - datamin[i]) / (datamax[i] - datamin[i])
            
def create_network(list_i_h_edges, list_h_h_edges, list_h_o_edges):
    num_nodes = num_input_nodes + num_hidden_nodes * num_hidden_layers + num_output_nodes
    network = list()
    for i in range(num_input_nodes):
        for j in range(num_input_nodes+1, num_input_nodes + num_hidden_nodes):
            network[i][j] = list_i_h_edges[i]
    stop_point = num_input_nodes + num_hidden_nodes + 1
    for i in range(num_input_nodes+1, num_input_nodes + num_hidden_nodes):
        for j in range(num_input_nodes + num_hidden_nodes + 1, num_input_nodes + num_hidden_nodes + num_hidden_nodes):
            network[i][j] = list_h_h_edges
    for i in range(num_input_nodes + num_hidden_nodes + 1, num_input_nodes + num_hidden_nodes + num_hidden_nodes):
        for j in range(num_input_nodes + num_hidden_nodes + num_hidden_nodes + 1, num_input_nodes + num_hidden_nodes + num_hidden_nodes + num_output_nodes):
            network[i][j] = list_h_o_edges
    return network
    
def evaluate_network(network):
    
    
#RUN
dataset = load_csv('2dData.tsv')
str_to_float(dataset)
print(dataset)
normalize_data(dataset)
print(dataset)