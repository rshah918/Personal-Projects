from random import random
from math import exp
#Implemented a neural network in python
#Uses Relu as the activation functions
#Play with the parameters and input dataset at the bottom of the code
def initialize_network(n_inputs, n_hidden, n_outputs):
		network = []
		hidden_layer = [{'weights': [1 for i in range(n_inputs+1)]} for j in range(n_hidden)]
		network.append(hidden_layer)
		output_layer = [{'weights': [1 for i in range(n_hidden+1)]} for j in range(n_outputs)]
		network.append(output_layer)
		return network

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

def transfer(activation):
	return max(0.1*activation,activation)

def transfer_deriv(output):
	if output <= 0:
		return .01
	else:
		return 1

def calculate_delta(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i == len(network)-1:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected - neuron['output'])
		else:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_deriv(neuron['output'])


def update_weights(network, inputs, learning_rate):
	for i in range(len(network)):
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
			for neuron in network[i]:
				for j in range(len(inputs)):
					neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
        neuron['weights'][-1] += learning_rate * neuron['delta']

def train_network(i, train, learning_rate, n_epoch, n_inputs, n_hidden, n_outputs):
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	for epoch in range(n_epoch):
		for data in train:
			row = [data[0]]
			output = forward_propagate(network, row)
			calculate_delta(network, data[-1])
			update_weights(network, row, learning_rate)
	return predict(network, [i])

def predict(network, test_input):
	 prediction = forward_propagate(network, test_input)
	 print prediction[-1]

def forward_propagate(network, inputs):
	inputs = inputs
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs
#Parameters to adjust
dataset = [[21.85, 24.94], [19.16,16.04], [21.87, 19.9], [21.81, 16.02], [21.21, 28.68], [20.71, 28.40], [22.4,15.94]
]

learning_rate = .001
n_epoch = 100
n_inputs = 1
n_hidden = 17
n_output = 1
#for i in range(0,14):
#	train_network(i, dataset, learning_rate, n_epoch, n_inputs, n_hidden, n_output)
#for i in dataset:
train_network(21.85, dataset, learning_rate, n_epoch, n_inputs, n_hidden, n_output)
