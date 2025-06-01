import random, math, sys
from utils import relu, relu_derivative, linear, linear_derivative, sigmoid, sigmoid_derivative

def softmax(logits):
	# Here we need to avoid overflow, if logit is large exp(logit) will be too large
	# We are only concerned with relativbe difference between logits
	# So we find the biggest logit and substract it to every other
	max_logit = max(logits)
	exp_logits = [math.exp(logit - max_logit) for logit in logits]
	exp_sum = sum(exp_logits) + 1e-15 # avoid dividing by zero

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

class MLP():

    def __init__(self, layers_sizes, activations_func):
        # minus 1 because layer_sizes[0] is the input layer (no activ func)
        assert((len(layers_sizes) - 1) == len(activations_func)) 

        self.loss_func = cross_entropy_loss 
        self.layers = [Layer(layers_sizes[i], layers_sizes[i+1], activations_func[i]) for i in range(len(layers_sizes) - 1)]

    def train(self, inputs, labels, epochs, batch_size, learning_rate):
        assert(len(inputs) == len(labels))

        for epoch in range(epochs):

            # Shuffle inputs between epoch to avoid different learning issues
            combined = list(zip(inputs, labels))
            random.shuffle(combined)
            shuffle_inputs, shuffle_labels = zip(*combined)

            for i in range(0, len(inputs), batch_size):

                batch_input = shuffle_inputs[i:i + batch_size]
                batch_label = shuffle_labels[i:i + batch_size]

                for layer in self.layers:
                    layer.reset_accumulate_gradients()

                total_loss = 0.0
                for input, label in zip(batch_input, batch_label):
                    # 1. Forward pass
                    activation = input
                    for layer in self.layers:
                        activation = layer.forward(activation)
                    
                    # 2. Compute Loss
                    loss, gradients = self.loss_func(activation, label)
                    total_loss += loss

                    # 3. Back propagation
                    for layer in reversed(self.layers):
                        gradients = layer.backward(gradients)
                        layer.accumulate_gradients()
                
                # 4. Update network params
                for layer in self.layers:
                    layer.update(learning_rate, len(batch_input))

                total_loss /= len(batch_input) 
                print(f"Epoch {epoch+1}/{epochs} - Batch {i/batch_size + 1}/{math.ceil(len(inputs)/batch_size)} -> Loss: {total_loss:.4f}")
        
    def evaluate(self, inputs, labels):
        correct = 0
        for input, label in zip(inputs, labels):
            activation = input
            for layer in self.layers:
                activation = layer.forward(activation)

            pred_idx = activation.index(max(activation)) 
            true_idx = label.index(max(label))
            
            if pred_idx == true_idx:
                correct += 1

        return 100 * correct / len(inputs)

class Layer():
    def __init__(self, nbin, nbneurons, activation_func):
        limit = math.sqrt(2/nbin)
        # Initialize weights and biases
        self.weights = [[random.uniform(-limit, limit) for _ in range(nbin)] for _ in range(nbneurons)] # Kaiming He initialization
        self.biases = [0.1 for _ in range(nbneurons)] # To avoid dying ReLU problem
        
        # Set activation function and it's derivate
        if activation_func == 'relu':
            self.activ_func = relu
            self.activ_func_deriv = relu_derivative 
        elif activation_func == 'linear':
            self.activ_func = linear
            self.activ_func_deriv = linear_derivative
        elif activation_func == 'sigmoid':
            self.activ_func = sigmoid 
            self.activ_func_deriv = sigmoid_derivative
        else:
            sys.exit(f"Unknown activation function {activation_func}")
    
    def forward(self, x):
        assert(len(x) == len(self.weights[0]))

        self.x = x # Keep track of inputs x needed later in backward pass 
        
        # We are computing: z = w*x + b ,for each neurons
        self.z = [sum(x[i] * self.weights[j][i] for i in range(len(x))) + self.biases[j] for j in range(len(self.weights))]

        # Apply activation function:  a = σ(z)
        self.a = [self.activ_func(z) for z in self.z]

        return self.a
    
    def backward(self, gradients):
        # Expect as many gradient as neurons in the layer
        assert(len(gradients) == len(self.weights))

        # Here we want ∂L/∂z = ∂L/∂a * ∂a/∂z
        self.grad_z = [grad * self.activ_func_deriv(z) for grad, z in zip(gradients, self.z)]
        
        # As many grad_z as neurons
        assert(len(self.grad_z) == len(self.weights))

        # Here we want ∂L/∂w = ∂L/∂z * ∂z/∂w  
        # And ∂z/∂w = x, because derivative of z = w*x + b w.r.t w is x
        self.grad_w = [[grad_z * x for x in self.x] for grad_z in self.grad_z]

        # Here we want ∂L/∂b = ∂L/∂z * ∂z/∂b  
        # were ∂z/∂b = 1, because derivative of: z = w*x + b, w.r.t b is equal to 1
        # i.e: ∂L/∂b = ∂L/∂z * 1 = ∂L/∂z
        self.grad_b = self.grad_z[:] # Shallow copy

        # Here we want ∂L/∂x = ∂L/∂z * ∂z/∂x
        # And ∂z/∂x = w, because derivative of: z = w*x + b, w.r.t x is equal to w

        # Here on we want to know the impact of x on L, because our inputs x_i is the output of a Neuron n_i in 
        # the Layer L-1 it affect every neuron in the layer L
        self.grad_x = [sum(self.grad_z[j] * self.weights[j][i] for j in range(len(self.weights))) for i in range(len(self.x))]

        assert(len(self.grad_x) == len(self.weights[0]))

        return self.grad_x
    
    def update(self, learning_rate, batch_size):
        # Update biases
        for i in range(len(self.biases)):
            self.biases[i] -= (self.grad_b_sum[i] / batch_size) * learning_rate

        # Update weights
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= (self.grad_w_sum[i][j] / batch_size) * learning_rate
    
    def reset_accumulate_gradients(self):
        self.grad_w_sum = [[0.0] * len(w) for w in self.weights] 
        self.grad_b_sum = [0.0] * len(self.biases)
         
    def accumulate_gradients(self):
        for i in range(len(self.grad_b)):
            self.grad_b_sum[i] += self.grad_b[i]
        
        for i in range(len(self.grad_w)):
            for j in range(len(self.grad_w[i])):
                self.grad_w_sum[i][j] += self.grad_w[i][j]