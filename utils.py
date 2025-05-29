import math
import numpy as np
    
def relu(x):
	return max(0, x)

def relu_derivative(x):
	return 1.0 if x > 0 else 0.0

def linear(x):
    return x

def linear_derivative(x):
    return 1

def sigmoid(x):
    """Sigmoid activation function"""
    return 1.0 / (1.0 + math.exp(-x)) if x > -700 else 0.0  # Prevent overflow

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)

# Numpy optimized functions

def np_relu(x: np.ndarray):
    return np.maximum(0, x)

def np_relu_derivate(x: np.ndarray):
    # (x > 0) : convert to boolean Array
    #  astype(int) : convert True to 1 and False to 0
    return (x > 0).astype(int)