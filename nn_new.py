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
	
	def backward(self, gradients):
		for layer in reversed(self.layers):
			for neuron, grad in zip(layer.neurons, gradients):
				neuron.backward(grad)

	def stochastic_train(self, inputs, labels):
		'''Stochastic gradient descent training function
		'''
		assert(len(inputs) == len(labels)) 

		for input, label in zip(inputs, labels):
			# 1. Forward Pass
			logits = self.forward(inputs)
			proba = softmax(logits)

			# 2. Loss Compute
			loss = categorical_cross_entropy(label, proba)
		
		 	# 3. Compute gradients
			gradients = [y_proba - y_true for y_proba, y_true in zip(proba, label)]

			# 4. Propagate gradient back to the network
			self.backward(gradients)

	# def batch_train(self, inputs, labels):

	# 	batch_loss = 0.0
	# 	batch_proba = []

	# 	batch_proba = [batch_proba[i] + proba[i] for i in range(proba)]
	# 	batch_loss += loss 

class Layer():
	def __init__(self, nbin: int, nbneurons: int, activ_func):
		self.neurons = [Neuron(nbin, activ_func) for _ in range(nbneurons)]
	
	def forward(self, inputs: list):
		return [n.forward(inputs) for n in self.neurons]

class Neuron():
	def __init__(self, nbin: int, activ_func):
		self.learning_rate = 0.01
		self.bias = random.random()
		self.weights = [random.random() for _ in range(nbin)]
		self.activation = activ_func 

	def __repr__(self):
		return f"Neuron({self.weights})"
	
	def forward(self, inputs):
		self.inputs = inputs
		self.z = sum([x*w for x,w in zip(inputs, self.weights)]) + self.bias 
		return self.activation(self.z)

	def backward(self, gradient):
		# Reminder: 
		# a = Activation output
		# z = Raw weigthed input (before activation) = wi*xi+ b

		# gradient = ∂L/∂a  

		# Impact of z on L => ∂L/∂z
		# Applying chaine rule: ∂L/∂z = ∂L/∂a * ∂a/∂z
		grad_z = gradient * relu_derivate(self.z)

		# Impact of w on L => ∂L/∂w
		# ∂L/∂w = ∂L/∂z * ∂z/∂w 
		# z = x*w + b so ∂z/∂w = x
		self.grad_w = [grad_z * x for x in self.inputs]
		
		# Impact of b on L => ∂L/∂b
	    # ∂L/∂b = ∂L/∂z * ∂z/∂b
		self.grad_b = grad_z

		# Impact of x on L =>  ∂L/∂x
		# ∂L/∂x = ∂L/∂x * ∂z/∂x
		grad_x = [grad_z * w for w in self.weights]

		return grad_x
	
	def update(self):
		for grad, w in zip(self.grad_w, self.weights):
			w -= grad * self.learning_rate
		
		self.bias -= self.grad_b * self.learning_rate
			
mlp = MLP([[3, 3], [3, 2]])
print(mlp.forward([1, 2, 3]))