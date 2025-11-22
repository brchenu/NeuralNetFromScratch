import numpy as np
from dataset import MNIST


def cross_entropy_loss(predicted: np.ndarray, actual: np.ndarray) -> float:
    grad = predicted - actual  # Gradient is y_pred - y_true

    epsilon = 1e-15
    predicted = np.clip(
        predicted, epsilon, 1 - epsilon
    )  # Clip to avoid log(0) or log(1)

    return -np.sum(actual * np.log(predicted)), grad


def iterate_regions(image: np.ndarray, ksize: int):
    assert image.ndim in (2, 3)
    rows, cols = image.shape

    print(f"{rows}x{cols}")

    for row in range(rows - ksize + 1):
        for col in range(cols - ksize + 1):
            yield image[row : row + ksize, col : col + ksize], row, col


class ConvLayer:
    def __init__(self, num_kernel: np.ndarray):
        self.kernel_size = 3
        self.num_kernel = num_kernel

        # Divide by 9 to avoid large initial values
        self.kernel = (
            np.random.randn(num_kernel, self.kernel_size, self.kernel_size) / 9
        )

    def forward(self, input: np.ndarray):

        out_w = input.shape[0] - self.kernel_size + 1
        out_h = input.shape[1] - self.kernel_size + 1

        output = np.zeros((out_w, out_h, self.num_kernel), dtype=input.dtype)

        for iter, row, col in iterate_regions(input, self.kernel_size):
            output[row, col] = np.sum(
                iter * self.kernel, axis=(1, 2)
            )  # maybe add clipping later

        return output
    
    def backward(self, grad, learning_rate):
        # grad shape: (out_w, out_h, num_kernel)
        dL_dK = np.zeros_like(self.kernel)

        for iter, row, col in iterate_regions(self.input, self.kernel_size):
            for k in range(self.num_kernel):
                dL_dK[k] += grad[row, col, k] * iter

        # Update kernels
        self.kernel -= learning_rate * dL_dK

        # Compute gradient w.r.t input if needed (not implemented here)
        return None  # Placeholder


class MaxPool:
    def forward(self, input: np.ndarray, psize: int):
        # Improve: allow to define stride
        self.input = input

        height, width, num_img = input.shape

        out = np.zeros((height // psize, width // psize, num_img), dtype=input.dtype)

        # Corresponds to the positions of max values in the input
        self.mask_max = np.zeros_like(input, dtype=bool)

        for row in range(0, height, psize):
            for col in range(0, width, psize):
                pool = input[row : row + psize, col : col + psize, :]

                # --- Keep track of max indices (FOR BACKPROP) ---
                reshaped_pool = pool.reshape(pool.shape[0] * pool.shape[1], pool.shape[2]) # Reshape to find max per channel 
                max_idx = np.argmax(reshaped_pool, axis=0) 
                coords = [np.unravel_index(i, (pool.shape[0], pool.shape[1])) for i in max_idx] # Get 2D coords
                    
                for idx, (x, y) in enumerate(coords):
                    # Shift to the correct position in the input array
                    x += row
                    y += col
                    self.mask_max[x, y, idx] = True  # Set to one the position of max value
                # -----------------------------------------------

                if pool.shape[0] < psize or pool.shape[1] < psize:
                    # If you can't fill the pool, skip it
                    continue
                out[row // psize, col // psize] = np.max(pool, axis=(0, 1))
        return out

    def backward(self, grad, learning_rate):
        print(f"grad shape: {grad.shape}")
        print(f"input shape: {self.input.shape}")
        print(f"max_idx shape: {self.mask_max.shape}")

        dX = np.zeros_like(self.input)
        grad_upsampled = grad.repeat(2, axis=0).repeat(2, axis=1)
        print(f"grad_upsampled shape: {grad_upsampled.shape}")

        dX[self.mask_max] = grad_upsampled[self.mask_max]

        return dX


class SoftmaxLayer:
    def __init__(self, in_dim, out_dim):
        self.weights = np.random.randn(
            in_dim, out_dim
        )  # Maybe divide or init differently
        self.biases = np.random.randn(out_dim)

    def forward(self, x: np.ndarray):
        self.x = x
        x = x.flatten()
        self.z = (x @ self.weights) + self.biases

        max = np.max(self.z)
        exp = np.exp(self.z - max)  # Shift for numerical stability

        return exp / np.sum(exp)

    def backward(self, grad, learning_rate):
        # grad shape : (out_dim, )
        x = self.x.flatten()

        self.grad_z = grad  # Since softmax + cross-entropy
        self.grad_w = np.outer(x, self.grad_z)  # (in_dim, out_dim)
        self.grad_b = self.grad_z  # (out_dim, )

        self.grad_x = self.weights @ self.grad_z  # (in_dim, )

        # Update weights and biases
        self.weights -= learning_rate * self.grad_w
        self.biases -= learning_rate * self.grad_b

        return self.grad_x.reshape(self.x.shape)


mnist = MNIST()
mnist.load()

train, labels = mnist.get_train_subset(1000, 1001)

img = np.array(train[0]).reshape(28, 28)

print(f"img: {img.shape} / {img.dtype}")


def print_image(image):
    for row in image:
        line = ""
        for pixel in row:
            if pixel > 0.5:
                line += "@"
            else:
                line += "."
        print(line)


print(f"label: {labels[0]}")
print_image(img)

conv_layer = ConvLayer(num_kernel=8)
pool_layer = MaxPool()
softmax_layer = SoftmaxLayer(in_dim=13 * 13 * 8, out_dim=10)

out = conv_layer.forward(img)  # 28x28x1 -> 26x26x8
print(f"output shape: {out.shape}")

out = pool_layer.forward(out, psize=2)  # 26x26x8 -> 13x13x8
print(f"pooled output shape: {out.shape}")

out = softmax_layer.forward(out)  # 13x13x8 -> 10
print(f"softmax output shape: {out.shape}")
print(f"output: {out}")

loss, grad = cross_entropy_loss(out, labels[0])

grad = softmax_layer.backward(grad, learning_rate=0.01)

pool_layer.backward(grad, learning_rate=0.01)

# Conv flow: 28x28x1 -> [Conv(3x3,8)] -> 26x26x8 -> [MaxPool(2x2)] -> 13x13x8 -> [Softmax] -> 10