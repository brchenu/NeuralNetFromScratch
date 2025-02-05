import math, random
import numpy as np

# Loss Function
def meanSquareError(ground_truth: list, prediction: list):
    assert(len(ground_truth) == len(prediction))

    errors = [math.pow(a - b, 2) for a, b in zip(ground_truth, prediction)]
    return sum(errors) / len(ground_truth)

def LossFunc(truth, pred):
    return 0.5 * (truth - pred)**2

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivate(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Neuron():
    def __init__(self, nin, bias: float = 0.0, weights = None):
        if (weights is None):
            self.weights = [random.uniform(0, 1) for _ in range(nin)]
        else:
            assert(nin == len(weights)) 
            self.weights = weights

        self.bias = bias
        self.gradients = []

    def forward(self, x):
        assert(len(x) == len(self.weights))
        self.inputs = x
        z1 = np.dot(self.weights, x) + self.bias

        return sigmoid(z1)
    
    def backward(self, out_grad):
        for x, w in zip(self.inputs, self.weights):
            z1 = x * w + self.bias
            dv_sig = sigmoid_derivate(z1)

            print(f"NN backward: {type(out_grad)} / {out_grad}")
            # x here because the local partial derivate of: x*w + b in by w is x
            gradient = x * dv_sig * out_grad
            self.gradients.append(gradient)

        return self.gradients

    def set_weights(self, custom_weights, bias):
        self.weights = custom_weights
        self.bias = bias
    
    def __repr__(self):
        return f"N(w={self.weights}, b={self.bias}, grad={self.gradients})"

class Layer():

    def __init__(self, nin, nneuron):
        self.neurons = [Neuron(nin) for _ in range(nneuron)]

    def forward(self, x):
        assert(len(x) == len(self.neurons[0].weights))
        outs = []
        for neuron in self.neurons:
            outs.append(neuron.forward(x))
    
        return outs

    def backward(self, out_grad):
        print(f"{len(out_grad)} / {len(self.neurons)}")
        assert(len(out_grad) == len(self.neurons))

        gradients = []
        for neuron, grad in zip(self.neurons, out_grad):
            gradients.append(neuron.backward(grad))

        print(f"layer output: {gradients}")

        return gradients

class MLP():

    def __init__(self, nin, layers: list):
        self.layers = []
        for idx, lsize in enumerate(layers):
            if idx == 0:
                self.layers.append(Layer(nin, lsize))
            else:
                self.layers.append(Layer(lsize - 1, lsize))
    
    def forward(self, x):
        input = x
        for layer in self.layers:
            input = layer.forward(input)

        return input

    def backward(self, out_grad):
        assert len(self.layers[-1].neurons) == len(out_grad)
        
        current_grad = out_grad
        for i,layer in enumerate(reversed(self.layers)):
            print(f"i:{i} | current grad: {current_grad}")
            current_grad = layer.backward(current_grad)

    def train(self, inputs, targets):
        assert(len(inputs) == len(targets))

        prediction = []
        for input in inputs:
            out = self.forward(input)
            prediction.append(out)

        print(f"prediction: {prediction}")
        print(f"targets: {targets}")

        loss = meanSquareError(targets[0], prediction[0])
        print(f"loss: {loss}")

        grads = []
        for t,p in zip(targets[0], prediction[0]):
            h = 0.00000001
            d = (LossFunc(t, p + h) - LossFunc(t, p)) / h
            
            grads.append(d)
            print(f"d: {d}")

        print(f"start back prop !")
        self.backward(grads)
    
inputs = np.array([0.05, 0.10])

# i1 = Neuron(1)
# i2 = Neuron(1)

h1 = Neuron(2, 0.35, np.array([0.15, 0.20]))
h2 = Neuron(2, 0.35, np.array([0.25, 0.30]))

o1 = Neuron(2, 0.60, np.array([0.40, 0.45]))
o2 = Neuron(2, 0.60, np.array([0.50, 0.55]))

h1_out = h1.forward(inputs)
h2_out = h2.forward(inputs)

new_in = np.array([h1_out, h2_out])

out1 = o1.forward(new_in)
out2 = o2.forward(new_in)

print(f"Model outputs: [{out1}, {out2}]")

predicted = np.array([out1, out2])
expected = np.array([0.01, 0.99])

l1 = LossFunc(expected[0], predicted[0])
l2 = LossFunc(expected[1], predicted[1])
print(f"l1:{l1}, l2:{l2}")

h = 0.00000001
d = (LossFunc(expected[0], predicted[0] + h) - LossFunc(expected[0], predicted[0])) / h
d2 = (LossFunc(expected[1], predicted[1] + h) - LossFunc(expected[1], predicted[1])) / h
print(f"d: {d} / d2: {d2}")

error = meanSquareError(expected, predicted)
print(error)

o1.backward(d)

print("\n=======================\n")

mlp = MLP(2, [2, 2])

mlp.layers[0].neurons[0].set_weights(np.array([0.15, 0.20]), 0.35)
mlp.layers[0].neurons[1].set_weights(np.array([0.25, 0.30]), 0.35)
print(mlp.layers[0].neurons)

mlp.layers[1].neurons[0].set_weights(np.array([0.40, 0.45]), 0.60)
mlp.layers[1].neurons[1].set_weights(np.array([0.50, 0.55]), 0.60)
print(mlp.layers[1].neurons)

mlp_out = mlp.forward(inputs)
print(f"MLP outputs: {mlp_out}")
print(f"MLP loss: {meanSquareError(expected, mlp_out)}")

h = 0.00000001
d = (LossFunc(expected[0], mlp_out[0] + h) - LossFunc(expected[0], mlp_out[0])) / h
print(f"MLP grad: {d}")


in2 = [inputs]
ex2 = [expected]
mlp.train(in2, ex2)

print(f"Neuron weights: {mlp.layers[0].neurons[0]}")