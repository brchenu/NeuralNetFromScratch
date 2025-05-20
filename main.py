from mlp import MLP, cross_entropy_loss
from dataset import MNIST

def getChar(value):
    density_line = "@%#*+=-:. "
    char_idx = (len(density_line) - 1) * value // 255; 
    return density_line[char_idx]

def print_image(image):
    line = ""
    for i, val in enumerate(image):
        line += getChar(val) 
        if (i + 1) % 28 == 0:
            print(line)
            line = ""

def to_list(label):
    label_lst = [0.0] * 10
    label_lst[label] = 1.0
    return label_lst

def to_one_hot_encoding(labels: list):
    return [to_list(label) for label in labels]

mnist = MNIST()
mnist.load()

labels = to_one_hot_encoding(mnist.labels)
test_labels = to_one_hot_encoding(mnist.test_labels)

# Normalize inputs
images = [[pixel / 255.0 for pixel in img] for img in mnist.images]
test_images = [[pixel / 255.0 for pixel in img] for img in mnist.test_images]

sub_img = images[:20000]
sub_label = labels[:20000]

shape = [784, 128, 10]
activations = ['relu', 'linear']

epochs = 3
batch_size = 32 
learning_rate = 0.01

mlp = MLP(shape, activations, cross_entropy_loss)

mlp.train(sub_img, sub_label, epochs, batch_size, learning_rate)

sub_test_img = test_images[:1000]
sub_test_label = test_labels[:1000]

print(f"success rate: {mlp.evaluate(sub_test_img, sub_test_label)}") 