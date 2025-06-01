import numpy as np
import random, math
from utils import np_relu, np_relu_derivate, linear, linear_derivative


def softmax(pred: np.ndarray):
    """Softmax: convert logits to proba distribution."""
    # Next two lines are used to avoid math.exp overflow due to a too big pred
    # To solve that we substract max value, because we're only interested in relative diff
    stable_pred = pred - np.max(pred, axis=1, keepdims=True)
    exp_pred = np.exp(stable_pred)
    sum_exp = np.sum(exp_pred, axis=1, keepdims=True)
    return exp_pred / sum_exp


def cross_entropy_loss(pred: np.ndarray, true: np.ndarray):
    """Categorical Cross Entropy (CCE) and Softmax has a loss function.
    The reason is that it's a good loss function for classification tasks.
    """
    proba = softmax(pred)  # shape: (batch_size, nb_classes)

    epsilon = 1e-15
    batch_size = pred.shape[0]

    # Clip to avoid log(0) and log(1)
    y_pred = np.clip(proba, epsilon, 1 - epsilon)

    loss = -np.sum(true * np.log(y_pred)) / batch_size

    gradients = y_pred - true  # MAYBE AVERAGE HERE LATTER ?

    return loss, gradients


class MLP:
    def __init__(self, shape, activations):
        assert (len(shape) - 1) == len(activations)

        self.layers = [
            Layer(shape[i - 1], shape[i], activations[i - 1])
            for i in range(1, len(shape))
        ]

    def forward(self, inputs):
        for l in self.layers:
            inputs = l.forward(inputs)
        return inputs

    def train(self, inputs, labels, epochs, batch_size, learning_rate):
        x_batches = np.array(
            [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
        )
        y_batches = np.array(
            [labels[i : i + batch_size] for i in range(0, len(labels), batch_size)]
        )

        for _ in range(epochs):
            perm = np.random.permutation(len(x_batches))

            x_shuffle = x_batches[perm]
            y_shuffle = y_batches[perm]

            for x, y_true in zip(x_shuffle, y_shuffle):
                # 1. Forward pass
                y_pred = self.forward(x)

                # 2. Loss
                loss, gradients = cross_entropy_loss(y_pred, y_true)

                # 3. Backpropagation
                for layer in reversed(self.layers):
                    gradients = layer.backward(gradients)

                # 4. Update weights
                # Before updating the weights wee need to
                for layer in self.layers:
                    layer.update_params(learning_rate, batch_size)

                print(f"loss: {loss}")

    def evaluate(self, inputs, labels):
        pred = self.forward(inputs)
        count = 0
        for y_pred, y_true in zip(pred, labels):
            if np.argmax(y_pred) == np.argmax(y_true):
                count += 1

        return (count * 100) / labels.shape[0] 


class Layer:
    def __init__(self, nbin: int, nbneurons: int, activation):
        limit = 2 / math.sqrt(nbin)  # Kaiming He weights init

        self.weights = np.array(
            [
                [random.uniform(-limit, limit) for _ in range(nbin)]
                for _ in range(nbneurons)
            ]
        )
        self.bias = np.array([0.1 for _ in range(nbneurons)])

        match activation:
            case "linear":
                self.activation = linear
                self.activation_derv = linear_derivative
            case "relu":
                self.activation = np_relu
                self.activation_derv = np_relu_derivate
            case _:
                raise Exception(f"Unknown activaton function: {activation}")

    def forward(self, inputs: np.ndarray):
        self.x = inputs  # (batch_size, in_dim)
        self.z = (self.x @ self.weights.T) + self.bias  # (batch_size, out_dim)
        self.a = self.activation(self.z)  # (batch_size, out_dim)
        return self.a

    def backward(self, grad):
        # grad shape: (batch_size, out_dim)

        # ∂L/∂z = ∂L/∂a * ∂a/∂z
        self.grad_z = grad * self.activation_derv(self.z)

        # ∂L/∂w = ∂L/∂z * ∂z/∂w
        self.grad_w = self.grad_z.T @ self.x  # (out_dim, in_dim)

        # ∂L/∂b = ∂L/∂z * ∂z/∂b
        # Since b is a constant its equals to: ∂L/∂z * 1 => ∂L/∂z
        # self.grad_b = np.sum(self.grad_z, axis=0, keepdims=True)  # (1, out_dim)
        self.grad_b = self.grad_z.copy()

        # ∂L/∂x = ∂L/∂z * ∂z/∂x
        self.grad_x = self.grad_z @ self.weights

        return self.grad_x

    def update_params(self, learning_rate, batch_size):
        # grad_w is already summed accross the batch during backward
        # process when doing: self.grad_z.T @ x
        # So here we just need to divide it by batch_size
        self.weights -= learning_rate * self.grad_w / batch_size

        # grad_b is not summed here it's dim is (batch_size, out_dim)
        # that why we call np.sum then divide by batch_size
        self.bias -= learning_rate * np.sum(self.grad_b, axis=0) / batch_size