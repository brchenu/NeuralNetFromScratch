from mlp import MLP, cross_entropy_loss
from dataset import MNIST

mnist = MNIST()
mnist.load()

shape = [784, 128, 10]
activations = ['relu', 'linear']

epochs = 3
batch_size = 32 
learning_rate = 0.01

mlp = MLP(shape, activations, cross_entropy_loss)

train_img, train_lbl = mnist.get_train_subset(0, 10_000)

mlp.train(train_img, train_lbl, epochs, batch_size, learning_rate)

test_img, test_lbl = mnist.get_test_subset(0, 1000)
print(f"success rate: {mlp.evaluate(test_img, test_lbl)}") 