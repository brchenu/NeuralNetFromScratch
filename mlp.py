import random, math, sys

def relu(x):
	return max(0, x)

def relu_derivative(x):
	return 1.0 if x > 0 else 0.0

def linear(x):
    return x

def linear_derivative(x):
    return 1

class MLP():

    def __init__(self, layers_sizes, activations_func, loss_func):
        # minus 1 because layer_sizes[0] is the input layer (no activ func)
        assert((len(layers_sizes) - 1) == len(activations_func)) 

        self.loss_func = loss_func
        self.layers = [Layer(layers_sizes[i], layers_sizes[i+1], activations_func[i]) for i in range(len(layers_sizes) - 1)]

class Layer():
    def __init__(self, nbin, nbneurons, activation_func):
        limit = math.sqrt(2/nbin)
        # Initialize weights and biases
        self.weights = [[random.uniform(0, limit) for _ in range(nbin)] for _ in range(nbneurons)] # Kaiming He initialization
        self.biases = [0.1 for _ in range(nbneurons)] # To avoid dying ReLU problem

        # Set activation function and it's derivate
        if activation_func == 'relu':
            self.activ_func = relu
            self.activ_func_deriv = relu_derivative 
        elif activation_func == 'linear':
            self.activ_func = linear
            self.activ_func_deriv = linear_derivative
        else:
            sys.abort(f"Unknown activation function {activation_func}")
    
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
        self.grad_b = self.grad_z[:] # copy of the entire list

        # Here we want ∂L/∂x = ∂L/∂z * ∂z/∂x
        # And ∂z/∂x = w, because derivative of: z = w*x + b, w.r.t x is equal to w

        # Here on we want to know the impact of x on L, because our inputs x_i is the output of a Neuron n_i in 
        # the Layer L-1 it affect every neuron in the layer L
        self.grad_x = [sum(self.grad_z[j] * self.weights[j][i] for j in range(len(self.weights))) for i in range(len(self.x))]

        assert(len(self.grad_x) == len(self.weights[0]))