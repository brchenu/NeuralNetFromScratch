import numpy as np
import math
from typing import Tuple


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
    def __init__(self, cin: int, cout: int, ksize: int, stride: int = 1):
        self.cin = cin
        self.cout = cout
        self.ksize = ksize
        self.stride = stride

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
                patch, self.kernels, axes=([0, 1, 2], [1, 2, 3])
            )

        return out

    def backward(self, grad: np.ndarray, learning_rate: float = 0.001):
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
        self.kernels -= learning_rate * dK

        return dX


class MaxPool2d:
    def __init__(self, size: int, stride: int):
        self.size = size
        self.stride = stride

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


class SoftmaxCrossEntropyLayer:
    def __init__(self, in_dim: int, out_dim: int):
        self.w = np.random.randn(in_dim, out_dim) / math.sqrt(in_dim)
        self.b = np.zeros(out_dim)

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        x: (in_dim,)
        y: (out_dim,) one-hot
        returns: scalar loss
        """
        self.x = x.flatten()
        self.y = y

        # Linear
        self.z = self.x @ self.w + self.b

        # Softmax (stable)
        exp = np.exp(self.z - np.max(self.z))
        self.probs = exp / np.sum(exp)

        # Cross-entropy loss
        loss = -np.sum(self.y * np.log(self.probs + 1e-9))
        return loss

    def backward(self, learning_rate: float = 0.001) -> np.ndarray:
        """
        returns: dX (gradient w.r.t input)
        """
        # Combined gradient
        dZ = self.probs - self.y  # (out_dim,)

        # Parameter gradients
        self.dW = np.outer(self.x, dZ)  # (in_dim, out_dim)
        self.dB = dZ  # (out_dim,)

        self.w -= learning_rate * self.dW
        self.b -= learning_rate * self.dB

        # Input gradient
        dX = dZ @ self.w.T  # (in_dim,)
        return dX


class ReLU:
    def forward(self, input: np.ndarray):
        self.input = input
        # Element-wise maximum between 0 and input
        return np.maximum(0, input)

    def backward(self, grad: np.ndarray, learning_rate):
        # ReLU has no learnable parameters, so we don't use learning_rate

        # Derivative of ReLU:
        # 1 if x > 0
        # 0 if x <= 0
        relu_grad = self.input > 0

        # Chain rule: multiply incoming gradient by local gradient
        return grad * relu_grad


a = np.full((16, 8, 3, 3), 1)
print(np.sum(a, axis=0))
