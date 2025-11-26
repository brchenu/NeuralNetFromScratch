import numpy as np
import math


def cross_entropy_loss(
    predicted: np.ndarray, actual: np.ndarray
) -> tuple[float, float]:
    grad = predicted - actual

    epsilon = 1e-15
    predicted = np.clip(predicted, epsilon, 1 - epsilon)

    return -np.sum(actual * np.log(predicted)), grad


def iterate_regions(input: np.ndarray, ksize: int):
    """Generate all possible ksize x ksize regions from a 3D array.
    @param arr: Input 3D numpy array (height x width x channels)
    @param ksize: Size of the square region to extract
    """

    assert input.ndim == 3
    rows, cols, _ = input.shape

    for row in range(rows - ksize + 1):
        for col in range(cols - ksize + 1):
            yield input[row : row + ksize, col : col + ksize, :], row, col


def Relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


class Conv2d:
    def __init__(self, cin: int, cout: int, ksize: int, pad: int = 0, stride: int = 1):
        self.cin = cin
        self.cout = cout
        self.ksize = ksize
        self.pad = pad
        self.stride = stride

        # Initialize kernels with small random values
        # Divide by (ksize * ksize) to avoid large initial values
        self.kernels = np.random.randn(cout, cin, ksize, ksize) / (ksize * ksize)

    def forward(self, input: np.ndarray):
        assert input.ndim == 3

        rows, cols, _ = input.shape

        out = np.zeros((rows - self.ksize + 1, cols - self.ksize + 1, self.cout))

        # patch shape: (ksize, ksize, cin)
        # kernel shape: (cout, cin, ksize, ksize)
        # out shape: (rows - ksize + 1, cols - ksize + 1, cout)
        for patch, row, col in iterate_regions(input, self.ksize):
            out[row, col, :] = np.sum(self.kernels * patch, axis=([1, 2, 3], [0, 1, 2]))

        return out

        # Equivalent using * and np.sum:
        # for iter, row, col in iterate_regions(input, 3):
        #     for k in range(kernel.shape[0]):
        #         kernel_slice = kernel[k].transpose(1, 2, 0)
        #         out2[row, col, k] = np.sum(kernel_slice * iter)


class MaxPool2d:
    def __init__(self, size: int, stride: int):
        self.size = size
        self.stride = stride

    def forward(self, input: np.ndarray):
        assert input.ndim == 3

        rows, cols, channels = input.shape
        out = np.zeros(
            (
                (rows - self.size) // self.stride + 1,
                (cols - self.size) // self.stride + 1,
                channels,
            )
        )

        for iter, row, col in iterate_regions(input, self.size):
            out[row, col, :] = np.amax(iter, axis=(0, 1))

        return out


class SoftmaxLayer:
    def __init__(self, in_dim: int, out_dim: int):
        # Divide by input_len to reduce the variance of initial values
        self.weights = np.random.randn(in_dim, out_dim) / math.sqrt(in_dim)
        self.biases = np.zeros(out_dim)

    def forward(self, input: np.ndarray):
        # Flatten input
        input_flat = input.flatten()
        self.input = input_flat

        self.z = input_flat @ self.weights + self.biases

        exp = np.exp(self.z - np.max(self.z))  # for numerical stability
        self.out = exp / np.sum(exp)

        return self.out
