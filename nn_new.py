import random, math

def relu(x):
	return max(0, x)

def relu_derivate(x):
	return 1.0 if x > 0 else 0.0
	
def categorical_cross_entropy(truth, pred):
	assert(len(truth) == len(pred))

	epsilon = 1e-15
	loss = 0.0
	nb_samples = len(truth)
	for i in range(nb_samples):
		pred = max(min(pred[i], 1 - epsilon), epsilon)
		loss += truth[i] * math.log(pred)
	
	return -loss / nb_samples

def softmax(logits):
	exp_logits = [math.exp(logit) for logit in logits]
	exp_sum = sum(exp_logits)
	return [exp_logit/exp_sum for exp_logit in exp_logits]

def batch_categorical_cross_entropy(truth, pred):
	assert(len(truth) == len(pred))

	total_loss = 0.0
	for t, p in zip(truth, pred):
		total_loss += categorical_cross_entropy(t, p)
		
	return total_loss

class MLP():
	def __init__(self, params: list, activ_func = relu):
		self.layers = [Layer(param[0], param[1], activ_func) for param in params]
	
	def forward(self, inputs):
		assert(len(inputs) == len(self.layers[0].neurons[0].weights))

		current_inputs = inputs
		for l in self.layers: 
			current_inputs = l.forward(current_inputs)
		
		return current_inputs
	
	# def backward(self, gradient):


	def stochastic_train(self, inputs, labels):
		'''Stochastic gradient descent training function
		'''
		assert(len(inputs) == len(labels)) 

		for i,l in zip(inputs, labels):
			# 1. Forward Pass
			predict = self.forward(i)
			# 2. Loss Compute
			loss = categorical_cross_entropy(l, i)

class Layer():
	def __init__(self, nbin: int, nbneurons: int, activ_func):
		self.neurons = [Neuron(nbin, activ_func) for _ in range(nbneurons)]
	
	def forward(self, inputs: list):
		return [n.forward(inputs) for n in self.neurons]

class Neuron():
	def __init__(self, nbin: int, activ_func):
		self.bias = random.random()
		self.weights = [random.random() for _ in range(nbin)]
		self.activation = activ_func 

	def __repr__(self):
		return f"Neuron({self.weights})"
	
	def forward(self, inputs):
		s1 = sum([x*w for x,w in zip(inputs, self.weights)]) + self.bias 
		return self.activation(s1)

	# def backward(self, gradient):
		 

mlp = MLP([[3, 3], [3, 2]])
print(mlp.forward([1, 2, 3]))