import math
import numpy as np
from dataset import MNIST


# --- Utility functions ---
def print_image(image):
    for row in image:
        line = ""
        for pixel in row:
            if pixel > 0.5:
                line += "@"
            else:
                line += "."
        print(line)


# --- End Utility functions ---


def cross_entropy_loss(predicted: np.ndarray, actual: np.ndarray) -> float:
    grad = predicted - actual  # Gradient is y_pred - y_true

    epsilon = 1e-15
    predicted = np.clip(
        predicted, epsilon, 1 - epsilon
    )  # Clip to avoid log(0) or log(1)

    return -np.sum(actual * np.log(predicted)), grad


def iterate_regions(image: np.ndarray, ksize: int):
    assert image.ndim == 3

    rows, cols, _ = image.shape
    print(f"image shape: {image.shape}")

    for row in range(rows - ksize + 1):
        for col in range(cols - ksize + 1):
            yield image[row : row + ksize, col : col + ksize, :], row, col


# Convolutional Layer
class Conv2d:
    def __init__(self, num_kernel: np.ndarray):
        self.kernel_size = 3
        self.num_kernel = num_kernel

        # Divide by 9 to avoid large initial values
        self.kernel = (
            np.random.randn(num_kernel, self.kernel_size, self.kernel_size) / 9
        )

    def forward(self, input: np.ndarray):
        self.input = input

        out_w = input.shape[0] - self.kernel_size + 1
        out_h = input.shape[1] - self.kernel_size + 1

        output = np.zeros((out_w, out_h, self.num_kernel), dtype=input.dtype)

        for iter, row, col in iterate_regions(input, self.kernel_size):
            output[row, col] = np.sum(
                iter * self.kernel, axis=(1, 2)
            )  # maybe add clipping later

        return output

    def backward(self, grad, learning_rate):
        # self.input shape: (in_w, in_h, num_kernel)
        # grad shape: (out_w, out_h, num_kernel)

        # ∂k/∂L = ∂out/∂L * ∂k/∂out
        dK = np.zeros_like(self.kernel)

        # ∂X/∂L = ∂out/∂L * ∂X/∂out
        dX = np.zeros_like(self.input)

        print(f"dX shape: {dX.shape}, grad shape: {grad.shape}")

        # Pad grad because its shape is smaller than input due to convolution
        padded_grad = np.pad(
            grad,
            pad_width=((1, 1), (1, 1), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        for iter, row, col in iterate_regions(self.input, self.kernel_size):
            dK += (
                padded_grad[row : row + self.kernel_size, col : col + self.kernel_size]
                * iter[:, :, np.newaxis]
            )

        # for iter, row, col in iterate_regions(self.input, self.kernel_size):
        #     for j in range(self.num_kernel):
        #         # ∂k/∂out = iter, becauase out = sum(iter * k), so derivative of iter * k w.r.t k is iter
        #         # Since k as a impact on multiple input, we sum over all positions
        #         dL_dK[j] += (
        #             padded_grad[
        #                 row : row + self.kernel_size, col : col + self.kernel_size, j
        #             ]
        #             * iter
        #         )

        #         # ∂x/∂out = k, because out = sum(iter * k), so derivative of iter * k w.r.t iter is k
        #         # Since X is impacted by multiple kernel, we sum over all kernel applications
        #         dX[row : row + self.kernel_size, col : col + self.kernel_size] += (
        #             padded_grad[row, col, j] * self.kernel[j]
        #         )

        # Update kernels
        self.kernel -= learning_rate * dK

        # Compute gradient w.r.t input if needed (not implemented here)
        return dX


# Max Pooling Layer
class MaxPool:
    def __init__(self, psize=2):
        self.psize = psize

    def forward(self, input: np.ndarray):
        # Improve: allow to define stride
        self.input = input

        height, width, num_img = input.shape

        out = np.zeros(
            (height // self.psize, width // self.psize, num_img), dtype=input.dtype
        )

        # Corresponds to the positions of max values in the input
        self.mask_max = np.zeros_like(input, dtype=bool)

        for row in range(0, height, self.psize):
            for col in range(0, width, self.psize):
                pool = input[row : row + self.psize, col : col + self.psize, :]

                # --- Keep track of max indices (FOR BACKPROP) ---
                reshaped_pool = pool.reshape(
                    pool.shape[0] * pool.shape[1], pool.shape[2]
                )  # Reshape to find max per channel
                max_idx = np.argmax(reshaped_pool, axis=0)
                coords = [
                    np.unravel_index(i, (pool.shape[0], pool.shape[1])) for i in max_idx
                ]  # Get 2D coords

                for idx, (x, y) in enumerate(coords):
                    # Shift to the correct position in the input array
                    x += row
                    y += col
                    self.mask_max[x, y, idx] = (
                        True  # Set to one the position of max value
                    )
                # -----------------------------------------------

                if pool.shape[0] < self.psize or pool.shape[1] < self.psize:
                    # If you can't fill the pool, skip it
                    continue
                out[row // self.psize, col // self.psize] = np.max(pool, axis=(0, 1))
        return out

    #  Dummy parameters for compatibility with other layers
    def backward(self, grad, _):
        # Important to note MaxPool has no learnable parameters,
        # so learning_rate and gradient accumulation are not used here
        dX = np.zeros_like(self.input)
        grad_upsampled = grad.repeat(self.psize, axis=0).repeat(self.psize, axis=1)

        dX[self.mask_max] = grad_upsampled[self.mask_max]

        print(f"dX --> shape: {dX.shape}")

        return dX


# Softmax layer combined with Cross-Entropy Loss
class Softmax:
    def __init__(self, in_dim, out_dim):
        self.weights = np.random.randn(in_dim, out_dim)
        self.biases = np.random.randn(out_dim)

    def forward(self, x: np.ndarray):
        self.x = x

        # Classic fully connected layer
        x = x.flatten()
        self.z = (x @ self.weights) + self.biases

        # Apply softmax
        max = np.max(self.z)
        exp = np.exp(self.z - max)  # Shift for numerical stability

        return exp / np.sum(exp)

    def backward(self, grad, learning_rate):
        # self.x shape : (in_dim, )
        # grad shape : (out_dim, )
        x = self.x.flatten()

        self.grad_z = grad  # Since softmax + cross-entropy
        self.grad_w = np.outer(x, self.grad_z)  # (in_dim, out_dim)
        self.grad_b = self.grad_z  # (out_dim, )

        self.grad_x = self.weights @ self.grad_z  # (in_dim, )

        # Update weights and biases
        self.weights -= learning_rate * self.grad_w
        self.biases -= learning_rate * self.grad_b

        # Reshape gradient to match input shape
        return self.grad_x.reshape(self.x.shape)


class CNN:
    def __init__(self, layers, learning_rate: float):
        self.layers = layers
        self.learning_rate = learning_rate

    def forward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.learning_rate)
        return grad

    def train(self, inputs, labels, epochs, batch_size):
        assert len(inputs) == len(labels)

        for epoch in range(epochs):
            # Shuffle inputs between epoch to avoid different learning issues
            combined = list(zip(inputs, labels))
            np.random.shuffle(combined)
            shuffle_inputs, shuffle_labels = zip(*combined)

            for i in range(0, len(inputs), batch_size):
                batch_input = shuffle_inputs[i : i + batch_size]
                batch_label = shuffle_labels[i : i + batch_size]

                total_loss = 0.0
                for input, label in zip(batch_input, batch_label):
                    # 1. Forward pass
                    activation = input
                    for layer in self.layers:
                        activation = layer.forward(activation)

                    # 2. Compute Loss
                    loss, gradients = cross_entropy_loss(activation, label)
                    total_loss += loss

                    # 3. Back propagation
                    self.backward(gradients)

                total_loss /= len(batch_input)
                print(
                    f"Epoch {epoch+1}/{epochs} - Batch {i/batch_size + 1}/{math.ceil(len(inputs)/batch_size)} -> Loss: {total_loss:.4f}"
                )


# mnist = MNIST()
# mnist.load()


# # Conv flow: 28x28x1 -> [Conv(3x3,8)] -> 26x26x8 -> [MaxPool(2x2)] -> 13x13x8 -> [Softmax] -> 10

# cnn = CNN(
#     layers=[Conv2d(num_kernel=8), MaxPool(), Softmax(in_dim=13 * 13 * 8, out_dim=10)],
#     learning_rate=0.01,
# )

# train, labels = mnist.get_train_subset(1000, 1200)

# train = [np.array(img).reshape(28, 28) for img in train]
# labels = [np.array(label) for label in labels]

# cnn.train(train, labels, epochs=3, batch_size=32)

a = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])
b = np.repeat(a[:, :, np.newaxis], 8, axis=2)
print(f"a shape: {a.shape}")

for iter, row, col in iterate_regions(a[:,:,np.newaxis], 3):
    print(f"iter shape: {iter.shape}, row: {row}, col: {col}")
