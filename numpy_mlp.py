import numpy as np
import random, math
from utils import np_relu, np_relu_derivate, linear, linear_derivative


def softmax(pred: np.ndarray):
    """Softmax: convert logits to proba distribution."""
    # Next two lines are used to avoid math.exp overflow due to a too big pred
    # To solve that we substract max value, because we're only interested in relative diff
    max_pred = pred.max()
    exp_pred = [math.exp(y - max_pred) for y in pred]
    return np.array([math.exp(y) / sum(exp_pred) for y in exp_pred])


def cross_entropy_loss(pred: np.ndarray, true: np.ndarray):
    """Categorical Cross Entropy (CCE) and Softmax has a loss function.
    The reason is that it's a good loss function for classification tasks.
    """
    proba = softmax(pred)
    epsilon = 1e-15

    gradients = []
    for y_pred, y_true in zip(proba, true):
        # Here we want to avoid log(0) or log(1)
        y_pred = max(min(y_pred, 1 - epsilon), epsilon)
        loss += y_true * math.log(y_pred + epsilon)

        # Calculate the gradient based on the derivate
        # of the Softmax + Cross Entropy, which result to: pred - true
        gradients.append(y_pred - y_true)

    return -loss, np.array(gradients)


class MLP:
    def __init__(self, shape, activations):
        self.layers = [
            Layer(shape[i - 1], shape[i], activations[i - 1][0], activations[i - 1][1])
            for i in range(1, len(shape))
        ]

    def forward(self, inputs):
        for l in self.layers:
            inputs = l.forward(inputs)
        return inputs

    def train(self, epochs, inputs, labels, batch_size, learning_rate):
        x_batches = np.array(
            [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
        )
        y_batches = np.array(
            [labels[i : i + batch_size] for i in range(0, len(labels), batch_size)]
        )

        print(f"x_batch shape: {x_batches.shape}")

        for _ in range(epochs):
            perm = np.random.permutation(len(x_batches))

            x_shuffle = x_batches[perm]
            y_shuffle = y_batches[perm]

            for x, y in zip(x_shuffle, y_shuffle):
                # 1. Forward pass
                pred = self.forward(x)

                # 2. Loss
                loss, gradients = cross_entropy_loss(x, y)

        #         # 3. Backpropagation

        #     # 4. Update weights

        # print(f"batches shape: {inputs.shape}")


class Layer:
    def __init__(self, nbin: int, nbneurons: int, activation, activation_deriv):
        limit = 2 / math.sqrt(nbin)  # Kaiming He weights init

        self.weights = np.array(
            [
                [random.uniform(-limit, limit) for _ in range(nbin)]
                for _ in range(nbneurons)
            ]
        )
        self.bias = np.array([0.1 for _ in range(nbneurons)])
        self.activation = activation
        self.activation_derv = activation_deriv

    def forward(self, inputs: np.ndarray):
        self.x = inputs                                 # (batch_size, in_dim)
        self.z = (self.x @ self.weights.T) + self.bias  # (batch_size, out_dim)
        self.a = self.activation(self.z)                # (batch_size, out_dim)
        return self.a

    def backward(self, grad):
        # grad shape: (batch_size, out_dim)

        # ∂L/∂z = ∂L/∂a * ∂a/∂z
        self.grad_z = grad * self.activation_derv(self.z)

        # print(f"shape grad_z: {self.grad_z.shape}")

        # ∂L/∂w = ∂L/∂z * ∂z/∂w
        self.grad_w = self.grad_z.T @ self.x  # (out_dim, in_dim)

        # ∂L/∂b = ∂L/∂z * ∂z/∂b 
        # Since b is a constant its equals to: ∂L/∂z * 1 => ∂L/∂z
        self.grad_b = np.sum(self.grad_z, axis=0, keepdims=True)  # (1, out_dim)

        # ∂L/∂x = ∂L/∂z * ∂z/∂x
        self.grad_x = self.grad_z @ self.weights

        return self.grad_x

    def reset_accumulated_gradients(self): 
        self.grad_w_accum = np.zeros(3, dtype=float)
        self.grad_b_accum = np.zeros(3, dtype=float)

random.seed(42)
inputs = np.array([[random.uniform(-1, 1) for _ in range(2)] for _ in range(100)])
print(f"inputs shape: {inputs.shape}")

shape = [2, 3, 3]
activations = [[np_relu, np_relu_derivate], [linear, linear_derivative]]

mlp = MLP(shape, activations)

epochs = 1
batch_size = 5
# mlp.train(epochs, inputs, inputs, batch_size,learning_rate=0.1)

# layer = Layer(2, 3, np_relu, np_relu_derivate)

# print(f"---------> inputs: {inputs.shape}")
# print(f"in: {inputs[0]}")
# print(f"l-forward: {layer.forward(inputs[0]).shape}")
# # print(f"l-backward: {layer.backward(np.array([[0.2, -0.3, 0.1], [1, 2, 3]]))}")
# print(f"l-backward: {layer.backward(np.array([0.2, -0.3, 0.1]))}")

# # test = np.array([[1, 2], [3, 4]])
# test = np.array([1, 2])
# # test2 = np.array([[1, 2, 3], [4, 5, 6]])
# test2 = np.array([1, 2, 3])
# print(f"test: {test.shape}")
# print(f"test2: {test2.shape}")

# # print(f"mul: {test2.reshape(-1, 1) * test}")
# print(f"mul: {test2.T @ test}")