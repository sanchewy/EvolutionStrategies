import sys
from csv import reader
import numpy as np
import random

class NeuralNetwork:
    #configuration parameters
    num_input_nodes = 5
    num_output_nodes = 1
    num_hidden_layers = 2
    num_hidden_nodes = 10         #per-layer
    data_set_location = "datasets/airfoil.txt"
    dataset = list()
    test_dataset = list()
    num_test_points = 20

    def __init__(self, databreak):
        dataset = self.load_csv(self.data_set_location)
        #print(dataset)
        self.str_to_float(dataset)
        self.normalize_data(dataset)
        holder = dataset
        length = int(len(holder)/5)
        stop = databreak * length
        self.test_dataset = holder[stop:stop+length]
        self.dataset = holder[:stop]+holder[stop+length:len(holder)]

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

    def forward_propagate(self, in_weights, h_weights, out_weights):
        sum_error = 0
        in_weights_array = np.array_split(np.array(in_weights), self.num_input_nodes)
        h_weights_array = np.array_split(np.array(h_weights), self.num_hidden_nodes)
        out_weights_array = np.array_split(np.array(out_weights), self.num_output_nodes)
    #    print("IN: "+str(len(in_weights_array))+" H: "+str(len(h_weights_array))+" Out:"+str(len(out_weights_array)))
    #    print("IN: "+str(len(in_weights_array[0]))+" H: "+str(len(h_weights_array[0]))+" Out:"+str(len(out_weights_array[0])))
        for i in range(0, self.num_test_points):
            test_index = random.randint(0, len(self.dataset)-1)
            data = np.array(self.dataset[test_index])
            input_to_hidden = list()
            for j in range(len(in_weights_array[0])):
                sum = 0
                for k in range(len(in_weights_array)):
                    sum += in_weights_array[k][j] * data[k]
                input_to_hidden.append(1 - (np.tanh(sum)**(2)))
            input_to_out = list()
            for j in range(len(h_weights_array[0])):
                sum = 0
                for k in range(len(h_weights_array)-1):
                    sum += h_weights_array[k][j] * input_to_hidden[k]
                sum += h_weights_array[j][len(h_weights_array)-1] * 1     #bias node
                input_to_out.append(1 - (np.tanh(sum)**(2)))
            output = 0
            for j in range(len(out_weights_array)):
                sum = 0
                for k in range(len(out_weights_array[j])-1):
                    sum += out_weights_array[j][k] * input_to_out[j]
                sum += out_weights_array[j][len(out_weights_array)-1] * 1     #bias node
                output += sum
            sum_error += (output - data[len(data)-1]) ** (2)
        return sum_error/self.num_test_points

    def final_eval(self, in_weights, h_weights, out_weights):
        sum_error = 0
        in_weights_array = np.array_split(np.array(in_weights), self.num_input_nodes)
        h_weights_array = np.array_split(np.array(h_weights), self.num_hidden_nodes)
        out_weights_array = np.array_split(np.array(out_weights), self.num_output_nodes)
    #    print("IN: "+str(len(in_weights_array))+" H: "+str(len(h_weights_array))+" Out:"+str(len(out_weights_array)))
    #    print("IN: "+str(len(in_weights_array[0]))+" H: "+str(len(h_weights_array[0]))+" Out:"+str(len(out_weights_array[0])))
        for i in range(len(self.test_dataset)):
            data = np.array(self.test_dataset[i])
            input_to_hidden = list()
            for j in range(len(in_weights_array[0])):
                sum = 0
                for k in range(len(in_weights_array)):
                    sum += in_weights_array[k][j] * data[k]
                input_to_hidden.append(1-(np.tanh(sum) ** (2)))
            input_to_out = list()
            for j in range(len(h_weights_array[0])):
                sum = 0
                for k in range(len(h_weights_array)-1):
                    sum += h_weights_array[j][k] * input_to_hidden[j]
                sum += h_weights_array[j][len(h_weights_array)-1] * 1     #bias node
                input_to_out.append(1-(np.tanh(sum) ** (2)))
            output = 0
            for j in range(len(out_weights_array)):
                sum = 0
                for k in range(len(out_weights_array[j])-1):
                    sum += out_weights_array[j][k] * input_to_out[j]
                sum += out_weights_array[j][len(out_weights_array)-1] * 1     #bias node
                output += sum
            sum_error += (output - data[len(data)-1]) ** (2)
        return sum_error/(len(self.test_dataset))

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
