import random, math

class MLP():
    def __init__(self, layers_sizes, activations_func, loss_func):
        # minus 1 because layer_sizes[0] is the input layer (no activ func)
        assert((len(layers_sizes) - 1) == len(activations_func)) 

        self.loss_func = loss_func
        self.layers = [Layer(layers_sizes[i], layers_sizes[i+1], activations_func[i]) for i in range(len(layers_sizes) - 1)]

class Layer():
    def __init__(self, nbin, nbneurons, activation_func):
        limit = math.sqrt(2/nbin)
        self.weights = [[random.uniform(0, limit) for _ in range(nbin)] for _ in range(nbneurons)] # He initialization
        self.biases = [0.1 for _ in range(nbneurons)] # To avoid dying ReLU problem
        self.activation_func = activation_func
    
    def forward(self, inputs):
        assert(len(inputs) == len(self.weights[0]))

        # We are computing z = w*x + b for each neurons
        self.z = [(inputs[i] * self.weights[j][i] + self.biases[i] for i in range(len(inputs))) for j in range(len(self.weights))]
        
mlp = MLP([12, 10], [0], 0)