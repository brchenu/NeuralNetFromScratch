import numpy as np
import math
from typing import Tuple
from dataset import MNIST


def he_init(shape: Tuple[int, ...], fan_in: int) -> np.ndarray:
    """He (Kaiming) initialization for weights.

    Args:
        shape: Shape of the weight tensor to create
        fan_in: Number of input connections (e.g., cin * ksize * ksize for conv)

    Returns:
        Initialized weight array with shape `shape`

    Reference:
        He et al., "Delving Deep into Rectifiers", ICCV 2015
        https://arxiv.org/abs/1502.01852
    """
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


def iterate_regions(input: np.ndarray, ksize: int, stride: int = 1):
    """Yield sliding ksize×ksize patches from a 3D array.

    input: (height, width, channels)
    Yields: (patch, row, col) where row/col are the *top-left* input indices.
    """

    assert input.ndim == 3
    assert ksize > 0 and stride > 0

    rows, cols, _ = input.shape
    assert rows >= ksize and cols >= ksize

    for row in range(0, rows - ksize + 1, stride):
        for col in range(0, cols - ksize + 1, stride):
            yield input[row : row + ksize, col : col + ksize, :], row, col


class Conv2d:
    def __init__(
        self,
        cin: int,
        cout: int,
        ksize: int,
        stride: int = 1,
        learning_rate: float = 0.001,
    ):
        self.cin = cin
        self.cout = cout
        self.ksize = ksize
        self.stride = stride
        self.learning_rate = learning_rate

        # He initialization: optimal for ReLU activations
        # fan_in = number of inputs to each neuron = cin * ksize * ksize
        fan_in = cin * ksize * ksize
        self.kernels = he_init((cout, cin, ksize, ksize), fan_in)

    def forward(self, input: np.ndarray):
        assert input.ndim == 3

        self.input = input

        rows, cols, _ = input.shape

        out = np.zeros(
            (
                (rows - self.ksize) // self.stride + 1,
                (cols - self.ksize) // self.stride + 1,
                self.cout,
            )
        )

        for patch, row, col in iterate_regions(input, self.ksize, self.stride):
            out[row // self.stride, col // self.stride, :] = np.tensordot(
                patch, self.kernels, axes=([0, 1, 2], [2, 3, 1])
            )

        return out

    def backward(self, grad: np.ndarray):
        # grad shape: (out_rows, out_cols, cout)

        dK = np.zeros_like(self.kernels)
        dX = np.zeros_like(self.input)

        for patch, row, col in iterate_regions(self.input, self.ksize, self.stride):
            out_row = row // self.stride
            out_col = col // self.stride

            for k in range(self.cout):
                # dK[k] shape: (cin, ksize, ksize)
                # patch shape: (ksize, ksize, cin)
                # So we need to transpose patch to (cin, ksize, ksize)
                dK[k] += grad[out_row, out_col, k] * patch.transpose(2, 0, 1)

                # dX shape : (in_rows, in_cols, cin)
                # kernels[k] shape: (cin, ksize, ksize)
                # So we need to add to the corresponding region in dX
                dX[row : row + self.ksize, col : col + self.ksize, :] += (
                    self.kernels[k].transpose(1, 2, 0) * grad[out_row, out_col, k]
                )

        # Update kernels
        self.kernels -= self.learning_rate * dK

        return dX


class MaxPool2d:
    def __init__(self, size: int, stride: int = None):
        self.size = size
        self.stride = stride if stride is not None else size

    def forward(self, input: np.ndarray):
        assert input.ndim == 3

        self.input = input

        rows, cols, channels = input.shape
        out = np.zeros(
            (
                (rows - self.size) // self.stride + 1,
                (cols - self.size) // self.stride + 1,
                channels,
            )
        )

        self.max_indices = {}

        # Iterate over all channels at once
        for patch, row, col in iterate_regions(input, self.size, self.stride):

            # Flatten because for argmax
            flat_patch = patch.reshape(-1, patch.shape[2])
            idx = np.argmax(flat_patch, axis=0)

            # Keep track of max indices for backward pass
            self.max_indices[(row, col)] = idx

            max_vals = flat_patch[idx, np.arange(flat_patch.shape[1])]
            out[row // self.stride, col // self.stride, :] = max_vals

        return out

    def backward(self, grad: np.ndarray):
        # grad shape: (out_rows, out_cols, channels)

        dX = np.zeros_like(self.input)

        for (row, col), idx in self.max_indices.items():
            # Convert flat index back to 2D index within the patch
            unraveled_idx = np.unravel_index(idx, (self.size, self.size))

            for channel in range(grad.shape[2]):
                dX[
                    # [0][channel] because unraveled_idx is a tuple of arrays
                    # where each array corresponds to a channel
                    row + unraveled_idx[0][channel],
                    col + unraveled_idx[1][channel],
                    channel,
                ] += grad[row // self.stride, col // self.stride, channel]

        return dX


class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        self.y = y
        # subtract max for numerical stability
        exp = np.exp(logits - np.max(logits))
        self.probs = exp / np.sum(exp)

        # Add small constant to avoid log(0)
        return -np.sum(y * np.log(self.probs + 1e-9))

    def backward(self):
        return self.probs - self.y  # Returns initial gradient


class ReLU:
    def forward(self, input: np.ndarray):
        self.input = input
        # Element-wise maximum between 0 and input
        return np.maximum(0, input)

    def backward(self, grad: np.ndarray):
        # ReLU has no learnable parameters, so we don't use learning_rate

        # Derivative of ReLU:
        # 1 if x > 0
        # 0 if x <= 0
        relu_grad = self.input > 0

        # Chain rule: multiply incoming gradient by local gradient
        return grad * relu_grad


class Flatten:
    def forward(self, input: np.ndarray):
        self.input_shape = input.shape
        return input.flatten()

    def backward(self, grad: np.ndarray):
        return grad.reshape(self.input_shape)


class Linear:
    def __init__(self, fan_in, fan_out, learning_rate=0.001):
        self.w = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)  # Xavier init
        self.b = np.zeros(fan_out)
        self.learning_rate = learning_rate

    def forward(self, x):
        # x shape: (fan_in,)
        self.x = x
        # output shape: (fan_out,)
        return x @ self.w + self.b

    def backward(self, grad):
        # grad shape: (fan_out,)

        self.dW = np.outer(
            self.x, grad
        )  # (fan_in,) outer (fan_out,) = (fan_in, fan_out)
        self.dB = grad  # (fan_out,)

        # Update weights
        self.w -= self.learning_rate * self.dW
        self.b -= self.learning_rate * self.dB

        # dX shape: (fan_in,) - gradient for previous layer
        return grad @ self.w.T  # (fan_out,) @ (fan_out, fan_in) = (fan_in,)


class CNN:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray):

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


mnist = MNIST()
mnist.load()

train_data, train_labels = mnist.get_train_subset(0, 1_000)

layers = [
    Conv2d(cin=1, cout=8, ksize=3, stride=1, learning_rate=0.001),
    ReLU(),
    MaxPool2d(size=2, stride=2),
    Conv2d(cin=8, cout=16, ksize=3, stride=1, learning_rate=0.001),
    ReLU(),
    MaxPool2d(size=2),
    Flatten(),
    Linear(fan_in=16 * 5 * 5, fan_out=10, learning_rate=0.001),
]

cnn = CNN(layers=layers)
loss_func = SoftmaxCrossEntropy()

for data, label in zip(train_data, train_labels):
    data_shaped = np.array(data).reshape(28, 28, 1)

    logits = cnn.forward(data_shaped)
    loss = loss_func.forward(logits, label)

    print(f"Loss: {loss}")

    grad = loss_func.backward()
    cnn.backward(grad)

test_data, test_labels = mnist.get_test_subset(0, 100)

score = 0
for data, label in zip(test_data, test_labels):
    data_shaped = np.array(data).reshape(28, 28, 1)
    logits = cnn.forward(data_shaped)
    predicted = np.argmax(logits)

    if predicted == np.argmax(label):
        score += 1

print(f"Test accuracy: {score} / 10")
