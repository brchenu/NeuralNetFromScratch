import numpy as np
import random, math
from utils import np_relu, linear

class MLP():
    def __init__(self, shape, activations):
        self.layers = [Layer(shape[i-1], shape[i], activations[i-1]) for i in range(1, len(shape))]

    def forward(self, inputs):
        for l in self.layers:
           inputs = l.forward(inputs)
        return inputs

class Layer():
    def __init__(self, nbin: int, nbneurons: int, activation):
        limit = 2/math.sqrt(nbin) # Kaiming He weights init

        self.weights = np.array([[random.uniform(-limit, limit) for _ in range(nbin)] for _ in range(nbneurons)])
        self.bias = np.array([0.1 for _ in range(nbneurons)])
        self.activation = activation
    
    def forward(self, inputs: np.ndarray):
        self.x = inputs
        self.z = (self.weights @ self.x) + self.bias
        self.a = self.activation(self.z)
        return self.a

random.seed(42)
inputs = np.array([random.uniform(-1, 1) for _ in range(2)])

shape = [2, 3, 3]
activations = [np_relu, linear]
mlp = MLP(shape, activations)

print(f"mlp forward: {mlp.forward(inputs)}")