import numpy as np
import matplotlib.pyplot as plt

class Neuron:
	def __init__(self, bias=0):
		self.bias = bias
		self.weights = []

	def calculate_output(self, inputs):		# calculates output by finding weighted sum of all the inputs and squashing it
		self.inputs = inputs
		self.output = self.squash_with_sigmoid(self.calculate_net())
		return self.output

	def squash_with_tanh(self, net):		# squashes net of the neuron using tanh function that returns [-1,1]
		return np.tanh(net)

	def squash_with_sigmoid(self, net):		# squashes net of the neuron using sigmoid function
		return self.sigmoid(net)

	def sigmoid(self, x):					# sigmoid function that returns [0,1]
		return (1 / (1 + np.exp(-x)))

	def calculate_net(self):				# calculates the net of the neuron
		net = 0.0
		for i in range(0, len(self.inputs)):
			net = net + self.inputs[i] * self.weights[i]
		net = net + self.bias
		return net

	def calculate_delta(self, desired_output):		# calculates delta, used only for the output layer neurons 
		return (desired_output - self.output) * (-1) * (1 - self.output * self.output)	

	def calculate_error(self, desired_output):
		return (desired_output - self.output) * (desired_output - self.output) / 2

# Layer of the network which consists of neurons
class Layer:
	def __init__(self, neuron_number, bias=0):
		self.bias = bias
		self.neurons = []
		for i in range(0, neuron_number):
			self.neurons.append(Neuron(self.bias))

	def initialize_weights(self, input_number):		# initializes weights of the neurons randomly with gaussian distribution with mean = 0 and standard deviation = 1
		for neuron in self.neurons:
			for i in range(0, input_number):
				neuron.weights.append(np.random.normal(0,1))

	def feed_forward(self, inputs):					# feed forward in the layer, find outputs off all the neurons in the layer
		outputs = []
		for neuron in self.neurons:
			outputs.append(neuron.calculate_output(inputs))
		return outputs


# Neural network which consists of neurons and layers
class Network:
	def __init__(self, input_neuron_number=3, hidden_neuron_number=4, output_neuron_number=1, hidden_bias=0, output_bias=0):
		self.alpha = 0.15

		self.hidden_layer = Layer(hidden_neuron_number, hidden_bias)		# hidden layer of the network
		self.hidden_layer.initialize_weights(input_neuron_number)			

		self.output_layer = Layer(output_neuron_number, output_bias)		# output layer of the network
		self.output_layer.initialize_weights(hidden_neuron_number)

	def feed_forward(self, inputs):
		return self.output_layer.feed_forward(self.hidden_layer.feed_forward(inputs))	# feed forward and give the resulting output 

	def backpropagate(self, inputs, desired_output):
		self.feed_forward(inputs)

		output_deltas = []
		for i in range(0, len(self.output_layer.neurons)):											# calculate output layer neurons' delta values
			output_deltas.append(self.output_layer.neurons[i].calculate_delta(desired_output))

		hidden_deltas = []
		for i in range(0, len(self.hidden_layer.neurons)):
			output_delta_weight_sum = 0.0
			for j in range(0, len(self.output_layer.neurons)):										# calculate hidden layer neurons' delta values
				output_delta_weight_sum += output_deltas[j] * self.output_layer.neurons[j].weights[i]
			hidden_deltas.append((1 - self.hidden_layer.neurons[i].output * self.hidden_layer.neurons[i].output) * output_delta_weight_sum)

		for i in range(0, len(self.output_layer.neurons)):					# update output layer neurons' weights
			for j in range(0, len(self.output_layer.neurons[i].weights)):
				self.output_layer.neurons[i].weights[j] -= self.alpha * output_deltas[i] * self.output_layer.neurons[i].inputs[j]

		for i in range(0, len(self.hidden_layer.neurons)):					# update hidden layer neurons' weights
			for j in range(0, len(self.hidden_layer.neurons[i].weights)):
				self.hidden_layer.neurons[i].weights[j] -= self.alpha * hidden_deltas[i] * self.hidden_layer.neurons[i].inputs[j]

	def train(self, inputs, desired_outputs):		# backpropagates for each input in the inputs array
		for i in range(0, len(inputs)):
			self.backpropagate(inputs[i], desired_outputs[i])

	def calculate_error(self, desired_output):		# calculates error of the network for the latest used input by taking average of all the output layer neurons' errors
		total_error = 0.0
		for i in range(0, len(self.output_layer.neurons)):
			total_error += self.output_layer.neurons[i].calculate_error(desired_output[i])
		return total_error / len(self.output_layer.neurons)


plt.figure(1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

network = Network(2,4,1)
inputs = [[1,1],[1,0],[0,1],[0,0]]
outputs = [0,1,1,0]
errors = []
iteration_number = []
for i in range(0, 1000):
	network.train(inputs, outputs)
	network.feed_forward([1,0])
	errors.append(network.calculate_error([1]))
	iteration_number.append(i)
plt.subplot(221)
plt.title("xor with 2 inputs")
plt.plot(iteration_number, errors)
plt.xlabel("training iterations")
plt.ylabel("error")

network = Network(3,4,1)
inputs = [[1,1,1],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
outputs = [0,0,1,1,1,1,0,0]
errors = []
iteration_number = []
for i in range(0, 1000):
	network.train(inputs, outputs)
	network.feed_forward([1,0,1])
	errors.append(network.calculate_error([1]))
	iteration_number.append(i)
plt.subplot(222)
plt.title("xor with 3 inputs of which 3rd has no effect")
plt.plot(iteration_number, errors)
plt.xlabel("training iterations")
plt.ylabel("error")

network = Network()
inputs = [[1,1,1],[1,1,0],[1,0,1],[0,1,1],[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
outputs = [1,0,0,0,1,1,1,0]
errors = []
iteration_number = []
for i in range(0, 1000):
	network.train(inputs, outputs)
	network.feed_forward([1,0,1])
	errors.append(network.calculate_error([0]))
	iteration_number.append(i)
plt.subplot(223)
plt.title("xor with 3 inputs")
plt.plot(iteration_number, errors)
plt.xlabel("training iterations")
plt.ylabel("error")
#print(network.feed_forward([1,0,1]))

network = Network(4,4,1)
inputs = [[1,1,1,1],[1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1],[1,1,0,0],[1,0,1,0],[0,1,1,0],[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]]
outputs = [1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0]
errors = []
iteration_number = []
for i in range(0, 1000):
	network.train(inputs, outputs)
	network.feed_forward([1,0,1,0])
	errors.append(network.calculate_error([0]))
	iteration_number.append(i)
plt.subplot(224)
plt.title("xor with 4 inputs of which 4th has no effect")
plt.plot(iteration_number, errors)
plt.xlabel("training iterations")
plt.ylabel("error")

plt.show()
"""
network = Network(4,4,1)
plt.figure(2)
inputs = [[1,1,1,1],[1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1],[1,1,0,0],[1,0,1,0],[0,1,1,0],[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]]
outputs = [0,0,0,1,1,0,1,1,1,1,0,1,1,0,0,0]
errors = []
iteration_number = []
for i in range(0, 1000):
	network.train(inputs, outputs)
	network.feed_forward([1,0,1,0])
	errors.append(network.calculate_error([1]))
	iteration_number.append(i)
plt.subplot(221)
plt.title("xor with 4 inputs of which 3rd and 4th has no effect")
plt.plot(iteration_number, errors)
plt.xlabel("training iterations")
plt.ylabel("error")
"""
network = Network(4,4,1)
plt.figure(2)
inputs = [[1,1,1,1],[1,1,1,0],[1,1,0,1],[1,0,1,1],[0,1,1,1],[1,1,0,0],[1,0,1,0],[0,1,1,0],[1,0,0,1],[0,1,0,1],[0,0,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]]
outputs = [0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0]
errors = []
iteration_number = []
for i in range(0, 1000):
	network.train(inputs, outputs)
	network.feed_forward([1,0,1,1])
	errors.append(network.calculate_error([1]))
	iteration_number.append(i)
#plt.subplot(222)
plt.title("xor of xor of first and last 2 inputs ((1 xor 2) xor (3 xor 4))")
plt.plot(iteration_number, errors)
plt.xlabel("training iterations")
plt.ylabel("error")

plt.show()
#print(network.feed_forward([1,0,1,0]))

