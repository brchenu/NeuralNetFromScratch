import math
    
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