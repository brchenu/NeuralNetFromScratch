import random

class MLP():
	def __init__(self, params: list):
		self.layers = [Layer(param[0], param[1]) for param in params]
	
	def forward(self, inputs):
		assert(len(inputs) == len(self.layers[0].neurons[0].weights))

class Layer():
	def __init__(self, nbin: int, nbneurons: int):
		self.neurons = [Neuron(nbin) for _ in range(nbneurons)]
	
	def forward(self, inputs: list):
		return [n.forward(inputs) for n in self.neurons]

class Neuron():
	def __init__(self, nbin: int):
		self.bias = random.random()
		self.weights = [random.random() for _ in range(nbin)]

	def __repr__(self):
		return f"Neuron({self.weights})"
	
	def forward(self, inputs):
		return sum([x*w for x,w in zip(inputs, self.weights)]) + self.bias 

mlp = MLP([[3, 3], [3, 2]])
mlp.forward([1, 2, 3])