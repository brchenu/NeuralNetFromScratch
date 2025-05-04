import random, math

def linear(x):
	return x

def linear_derivate(x):
	return 1.0

def relu(x):
	return max(0, x)

def relu_derivate(x):
	return 1.0 if x > 0 else 0.0

def softmax(logits):
	# Here we need to avoid overflow, if logit is large exp(logit) will be to large
	# We are only concerned with relativbe difference between logits
	# So we find the biggest logit and substract it to every other
	max_logit = max(logits)
	exp_logits = [math.exp(logit - max_logit) for logit in logits]
	exp_sum = sum(exp_logits)

	return [exp_logit/exp_sum for exp_logit in exp_logits]

def cross_entropy_loss(logits, label):
	'''Categorical Cross Entropy (CCE) and Softmax combined loss function '''
	assert(len(logits)== len(label))

	pred = softmax(logits) # Convert logits to probabilistic prediction
	epsilon = 1e-15 # Here to avoid log(0)

	loss = 0.0
	gradients = []

	# Compute directly loss and gradients
	for y_true, y_pred in zip(label, pred):
		p = max(min(y_pred, 1 - epsilon), epsilon) # Clamp y_preed to avoid log(0)
		loss += -(y_true * math.log(p))

		# Derivate of Loss w.r.t logits (softmax + CE derivate)
		gradients.append(y_pred - y_true)

	return loss, gradients

def mse_loss(predictions, targets):
    assert len(predictions) == len(targets)

    loss = 0.0
    gradients = []

    for y_pred, y_true in zip(predictions, targets):
        diff = y_pred - y_true
        loss += diff ** 2
        gradients.append(2 * diff)  # Derivative of (y_pred - y_true)^2

    loss /= len(predictions)
    gradients = [g / len(predictions) for g in gradients] 

    return loss, gradients

def get_batches(inputs, labels, batch_size):
	''' Utility function to split inputs and in smaller batches'''
	assert(len(inputs) == len(labels))
	for i in range(0, len(inputs), batch_size):
		yield inputs[i:i+batch_size], labels[i:i+batch_size]

class MLP():
	def __init__(self, params: list, loss_function):
		self.layers = [Layer(param[0], param[1], param[2]) for param in params]
		self.loss_function = loss_function
	
	def forward(self, inputs):
		assert(len(inputs) == len(self.layers[0].neurons[0].weights))

		current_inputs = inputs
		for l in self.layers: 
			current_inputs = l.forward(current_inputs)
		
		return current_inputs
	
	def backward(self, gradients):
		curr_grads = gradients
		for layer in reversed(self.layers):
			curr_grads = layer.backward(curr_grads)

	def update(self, learning_rate):
		for l in self.layers:
			for neuron in l.neurons:
				neuron.update(learning_rate)

	def stochastic_train(self, inputs, labels, learning_rate=0.1):
		'''Stochastic gradient descent training function'''
		assert(len(inputs) == len(labels)) 

		for input, label in zip(inputs, labels):
			# 1. Forward Pass
			logits = self.forward(input)

		 	# 2. Compute Loss and gradients
			loss, gradients = self.loss_function(logits, label)

			print(f"loss: {loss}")

			# 3. Propagate gradient back to the network
			self.backward(gradients)

			# 4. Update networks parameters
			self.update(learning_rate)
		
	def minibatches_train(self, inputs, labels, batch_size, learning_rate=0.1): 
		assert(len(inputs) == len(labels))
		
		for batch_in, batch_lab in get_batches(inputs, labels, batch_size):
			total_loss = 0
			accumulated_gradients = []

			for input, label in zip(batch_in, batch_lab):
				# 1. Forward pass
				logits = self.forward(input)
				
				# 2. Compute Loss and gradients
				loss, gradients = self.loss_function(logits, label)

				total_loss += loss
				accumulated_gradients.append(gradients)
			
			# 3. Average the Loss and gradients over the entire batch
			total_loss /= len(batch_in)
			avg_gradients = [sum(grad[i] for grad in accumulated_gradients) / len(batch_in)
                         for i in range(len(accumulated_gradients[0]))]	
			print(f"Batch loss: {total_loss}")
			
			# 4. Backprogration
			self.backward(avg_gradients)

			# 5. Update parameters
			self.update(learning_rate)

	def batch_train(self, inputs, labels, learning_rate=0.1):
		total_loss = 0
		accumulate_gradient = []

		for input, label in zip(inputs, labels):
			# 1. Forward pass
			logits = self.forward(input)

			# 2. Compute Loss and graidents
			loss, gradients = self.loss_function(logits, label)

			total_loss += loss
			accumulate_gradient.append(gradients)

		# 3. Average loss and gradients over the entire batch
		total_loss /= len(inputs)
		avg_gradients = [sum(grad[i] for grad in accumulate_gradient) / len(inputs) for i in range(len(accumulate_gradient[0]))]
		print(f"Batch loss: {total_loss}")
	
		self.backward(avg_gradients)
		self.update(learning_rate)

class Layer():
	def __init__(self, nbin: int, nbneurons: int, activ_func):
		assert(len(activ_func) == 2)
		self.neurons = [Neuron(nbin, activ_func) for _ in range(nbneurons)]
	
	def forward(self, inputs: list):
		return [n.forward(inputs) for n in self.neurons]
	
	def backward(self, gradients):
		assert(len(gradients) == len(self.neurons))

		# In this function we need to sum the gradients returned by our different neurons for the 
		# same inputs because our inputs which is the output of a Neuron in the Layer L-1 affect every 
		# neuron in the layer L
        #
        # Ex: If I have 2 Neurons, and 3 inputs (x1, x2, x3), each neurons backward will return [x1_grad, x2_grad, x3_grad]
        # So we need to sum the x1_grad of the neuron N1 with the x1_grad of the neuron N2 in order to have
        # the entire impact of x1 before propagating it back to the previous layer.
        #
        # The reason we do that as mentionned above is because if we want to know the impact of x1 on the Loss 
        # we need to sum up all of its contribution.

		summed_out = [0.0] * len(self.neurons[0].weights)
		for neuron, grad in zip(self.neurons, gradients):
			out = neuron.backward(grad)
			summed_out = [s + o for s,o in zip(summed_out, out)]

		return summed_out

class Neuron():
	def __init__(self, nbin: int, activ_func):
		assert(len(activ_func) == 2)

		self.bias = random.uniform(-1, 1)
		self.weights = [random.uniform(-1, 1) for _ in range(nbin)]
		self.activation = activ_func[0] 
		self.activation_deriv = activ_func[1]

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
		grad_z =  gradient * self.activation_deriv(self.z) 

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
	
	def update(self, learning_rate):
		for i in range(len(self.weights)):
			self.weights[i] -= self.grad_w[i] * learning_rate
		
		self.bias -= self.grad_b * learning_rate
			
linear_func = (linear, linear_derivate)
relu_func = (relu, relu_derivate)

# Accucracy function use for classifiations tasks 
def evaluate_accuracy(model, inputs, labels):
    correct = 0
    for x, y in zip(inputs, labels):
        output = softmax(model.forward(x))
        predicted = output.index(max(output))
        actual = y.index(max(y))
        if predicted == actual:
            correct += 1
    return correct / len(inputs)


# Find the output range manually
f_min = 5
f_max = 7

# Normalized version
def normalize(y):
    return (y - f_min) / (f_max - f_min)

def denormalize(y):
    return y * (f_max - f_min) + f_min

f  = lambda a : 2*(a**2) + 5
n = 10000

inputs = [[random.uniform(-1, 1)] for _ in range(n)]
labels = [[normalize(f(x[0]))] for x in inputs]

mlp = MLP([[1, 16, relu_func], [16, 8, relu_func], [8, 1, linear_func]], mse_loss)

epochs = 30 

for _ in range(epochs):
	mlp.minibatches_train(inputs, labels, batch_size=1, learning_rate=0.001)

# for _ in range(epochs):
	# mlp.stochastic_train(inputs, labels, learning_rate=0.001)

test_data = [[random.uniform(-1, 1)] for _ in range(100)]
test_outputs = [[f(v[0])] for v in test_data]

success_rate = 1
for x, y in zip(test_data, test_outputs):
	pred = mlp.forward(x)
	print(f"diff: {abs(y[0] - denormalize(pred[0]))}")
	if (abs(y[0] - denormalize(pred[0])) <= 0.05):
		success_rate += 1

print(f"Success rate: {success_rate}")

print(f"forward: {denormalize(mlp.forward([0.5])[0])} : {f(0.5)}")
print(f"forward: {denormalize(mlp.forward([0.3])[0])} : {f(0.3)}")
print(f"forward: {denormalize(mlp.forward([-0.85])[0])} : {f(-0.85)}")