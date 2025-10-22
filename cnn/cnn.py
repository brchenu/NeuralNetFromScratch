import numpy as np
from dataset import MNIST


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


class MaxPool:
    @staticmethod
    def forward(input: np.ndarray, psize: int):
        # Improve: allow to define stride

        height, width, num_img = input.shape

        out = np.zeros((height // psize, width // psize, num_img), dtype=input.dtype)

        for row in range(0, height, psize):
            for col in range(0, width, psize):
                pool = input[row : row + psize, col : col + psize, :]
                if pool.shape[0] < psize or pool.shape[1] < psize:
                    # If you can't fill the pool, skip it
                    continue
                out[row // psize, col // psize] = np.max(pool, axis=(0, 1))
        return out


class SoftmaxLayer():
    def __init__(self, in_dim, out_dim):
        self.weights = np.random.randn(in_dim, out_dim) # Maybe divide or init differently
        self.biases = np.random.randn(out_dim)

    def forward(self, input: np.ndarray):
        max = np.max(input)
        exp = np.exp(input - max)

        return exp / np.sum(exp, axis=(1))

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

out = conv_layer.forward(img)
print(f"output shape: {out.shape}")

out = pool_layer.forward(out, psize=2)
print(f"pooled output shape: {out.shape}")
