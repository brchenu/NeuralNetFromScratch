import numpy as np
import math
from typing import Tuple
from dataset import MNIST


def he_init(shape: Tuple[int, ...], fan_in: int) -> np.ndarray:
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


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

        fan_in = cin * ksize * ksize
        self.kernels = he_init((cout, cin, ksize, ksize), fan_in)
        self.bias = np.zeros(cout)  # Add bias!

    def forward(self, input: np.ndarray):
        assert input.ndim == 4

        self.input = input
        batch_size, rows, cols, _ = input.shape

        out_h = (rows - self.ksize) // self.stride + 1
        out_w = (cols - self.ksize) // self.stride + 1

        self.col = self._im2col(input)
        kernels_flat = self.kernels.reshape(self.cout, -1)

        out = self.col @ kernels_flat.T  # (batch, out_h*out_w, cout)

        out = out + self.bias  # Add bias

        out = out.reshape(batch_size, out_h, out_w, self.cout)

        return out

    def _im2col(self, input):
        batch_size, rows, cols, cin = input.shape
        out_h = (rows - self.ksize) // self.stride + 1
        out_w = (cols - self.ksize) // self.stride + 1

        col = np.zeros((batch_size, out_h * out_w, cin * self.ksize * self.ksize))

        for i, row in enumerate(range(0, rows - self.ksize + 1, self.stride)):
            for j, col_idx in enumerate(range(0, cols - self.ksize + 1, self.stride)):
                patch = input[
                    :, row : row + self.ksize, col_idx : col_idx + self.ksize, :
                ]
                col[:, i * out_w + j, :] = patch.reshape(batch_size, -1)

        return col

    def backward(self, grad: np.ndarray):
        batch_size = grad.shape[0]

        grad_flat = grad.reshape(batch_size, -1, self.cout)

        # Gradient for kernels
        dK_flat = np.tensordot(self.col, grad_flat, axes=([0, 1], [0, 1]))
        self.dK = dK_flat.T.reshape(self.kernels.shape)

        # Gradient for bias
        self.dB = np.sum(grad_flat, axis=(0, 1))

        # Gradient for input
        kernels_flat = self.kernels.reshape(self.cout, -1)
        dcol = grad_flat @ kernels_flat

        dX = self._col2im(dcol)

        # Update parameters
        self.kernels -= self.learning_rate * self.dK / batch_size
        self.bias -= self.learning_rate * self.dB / batch_size

        return dX

    def _col2im(self, col):
        batch_size, _, rows, cols = self.input.shape
        out_h = (rows - self.ksize) // self.stride + 1
        out_w = (cols - self.ksize) // self.stride + 1

        dX = np.zeros_like(self.input)

        for i, row in enumerate(range(0, rows - self.ksize + 1, self.stride)):
            for j, col_idx in enumerate(range(0, cols - self.ksize + 1, self.stride)):
                patch = col[:, i * out_w + j, :].reshape(
                    batch_size, self.ksize, self.ksize, self.cin
                )
                dX[
                    :, row : row + self.ksize, col_idx : col_idx + self.ksize, :
                ] += patch

        return dX


class MaxPool2d:
    def __init__(self, size: int, stride: int = None):
        self.size = size
        self.stride = stride if stride is not None else size

    def forward(self, input: np.ndarray):
        # input shape: (batch, height, width, channels)
        assert input.ndim == 4

        self.input = input
        batch_size, rows, cols, channels = input.shape

        out_h = (rows - self.size) // self.stride + 1
        out_w = (cols - self.size) // self.stride + 1

        out = np.zeros((batch_size, out_h, out_w, channels))
        self.max_indices = np.zeros((batch_size, out_h, out_w, channels), dtype=int)

        for i, row in enumerate(range(0, rows - self.size + 1, self.stride)):
            for j, col_idx in enumerate(range(0, cols - self.size + 1, self.stride)):
                patch = input[
                    :, row : row + self.size, col_idx : col_idx + self.size, :
                ]
                patch_flat = patch.reshape(batch_size, -1, channels)

                idx = np.argmax(patch_flat, axis=1)  # (batch, channels)
                self.max_indices[:, i, j, :] = idx

                out[:, i, j, :] = np.take_along_axis(
                    patch_flat, idx[:, np.newaxis, :], axis=1
                ).squeeze(1)

        return out

    def backward(self, grad: np.ndarray):
        # grad shape: (batch, out_h, out_w, channels)
        batch_size, out_h, out_w, channels = grad.shape
        dX = np.zeros_like(self.input)
        rows = self.input.shape[1]
        cols = self.input.shape[2]

        for i, row in enumerate(range(0, rows - self.size + 1, self.stride)):
            for j, col_idx in enumerate(range(0, cols - self.size + 1, self.stride)):
                idx = self.max_indices[:, i, j, :]  # (batch, channels)
                idx_row, idx_col = np.unravel_index(idx, (self.size, self.size))

                for b in range(batch_size):
                    for c in range(channels):
                        dX[b, row + idx_row[b, c], col_idx + idx_col[b, c], c] += grad[
                            b, i, j, c
                        ]

        return dX


class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        self.y = y
        self.batch_size = y.shape[0]

        # Subtract max for numerical stability
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        # Average loss over batch
        return -np.mean(np.sum(y * np.log(self.probs + 1e-9), axis=1))

    def backward(self):
        # Return raw gradient, layers will handle batch averaging
        return self.probs - self.y


class ReLU:
    def forward(self, input: np.ndarray):
        self.input = input
        # Element-wise maximum between 0 and input
        return np.maximum(0, input)

    def backward(self, grad: np.ndarray):
        return grad * (self.input > 0)


class Flatten:
    def forward(self, input: np.ndarray):
        self.input_shape = input.shape
        return input.reshape(self.input_shape[0], -1)

    def backward(self, grad: np.ndarray):
        return grad.reshape(self.input_shape)


class Linear:
    def __init__(self, fan_in, fan_out, learning_rate=0.001, clip_value=1.0):
        self.w = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
        self.b = np.zeros(fan_out)
        self.learning_rate = learning_rate
        self.clip_value = clip_value

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad):
        batch_size = grad.shape[0]

        # Average gradients over batch
        self.dW = self.x.T @ grad / batch_size
        self.dB = np.sum(grad, axis=0) / batch_size

        # Gradient clipping to prevent overflow
        self.dW = np.clip(self.dW, -self.clip_value, self.clip_value)
        self.dB = np.clip(self.dB, -self.clip_value, self.clip_value)

        self.w -= self.learning_rate * self.dW
        self.b -= self.learning_rate * self.dB

        return grad @ self.w.T  # Don't divide dX, just pass through


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


# ============ TRAINING ============

mnist = MNIST()
mnist.load()


def create_batches(data, labels, batch_size):
    indices = np.random.permutation(len(data))
    for i in range(0, len(data), batch_size):
        batch_idx = indices[i : i + batch_size]
        yield np.array([data[j] for j in batch_idx]), np.array(
            [labels[j] for j in batch_idx]
        )


LEARNING_RATE = 0.006

layers = [
    Conv2d(cin=1, cout=8, ksize=3, stride=1, learning_rate=LEARNING_RATE),
    ReLU(),
    MaxPool2d(size=2, stride=2),
    Conv2d(cin=8, cout=16, ksize=3, stride=1, learning_rate=LEARNING_RATE),
    ReLU(),
    MaxPool2d(size=2),
    Flatten(),
    Linear(fan_in=16 * 5 * 5, fan_out=10, learning_rate=LEARNING_RATE),
]

cnn = CNN(layers=layers)
loss_func = SoftmaxCrossEntropy()

TRAIN_SIZE = 60_000
BATCH_SIZE = 32
EPOCHS = 3

# /!\ Be CAREFUL data are already normalized in dataset.py /!\
train_data, train_labels = mnist.get_train_subset(0, TRAIN_SIZE)

print("Starting training...")
print(f"Train size: {TRAIN_SIZE}, Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")

# Debug: Check first batch
for batch_data, batch_labels in create_batches(train_data, train_labels, BATCH_SIZE):
    data_shaped = np.array(batch_data).reshape(
        -1, 28, 28, 1
    )  # Already normalized in dataset.py

    print(f"\n=== DEBUG: First batch ===")
    print(
        f"Input shape: {data_shaped.shape}, range: [{data_shaped.min():.3f}, {data_shaped.max():.3f}]"
    )

    logits = cnn.forward(data_shaped)
    print(
        f"Logits shape: {logits.shape}, range: [{logits.min():.3f}, {logits.max():.3f}]"
    )

    loss = loss_func.forward(logits, batch_labels)
    print(f"Loss: {loss:.4f}")
    print(f"Probs sample: {loss_func.probs[0]}")

    grad = loss_func.backward()
    print(f"Initial grad shape: {grad.shape}, magnitude: {np.abs(grad).mean():.6f}")

    # Trace backward through each layer
    for i, layer in enumerate(reversed(cnn.layers)):
        grad = layer.backward(grad)
        layer_name = type(layer).__name__
        print(f"After {layer_name}: grad magnitude: {np.abs(grad).mean():.6f}")

    break  # Only first batch

print("\n=== Starting actual training ===\n")

for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0

    for batch_data, batch_labels in create_batches(
        train_data, train_labels, BATCH_SIZE
    ):
        data_shaped = np.array(batch_data).reshape(-1, 28, 28, 1)

        logits = cnn.forward(data_shaped)
        loss = loss_func.forward(logits, batch_labels)
        total_loss += loss
        num_batches += 1

        grad = loss_func.backward()
        cnn.backward(grad)

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

# Testing
print("\nTesting...")
test_data, test_labels = mnist.get_test_subset(0, 1000)

correct = 0
total = 0

for batch_data, batch_labels in create_batches(test_data, test_labels, BATCH_SIZE):
    batch_shaped = np.array(batch_data).reshape(-1, 28, 28, 1)
    logits = cnn.forward(batch_shaped)
    predictions = np.argmax(logits, axis=1)
    targets = np.argmax(batch_labels, axis=1)
    correct += np.sum(predictions == targets)
    total += len(batch_labels)

print(f"Test accuracy: {correct}/{total} = {100 * correct / total:.2f}%")
