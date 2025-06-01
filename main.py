import argparse
import numpy as np
from mlp import MLP
from numpy_mlp import MLP as NumpyMLP
from dataset import MNIST

def load_data(model_type):

    mnist = MNIST()
    mnist.load()
    
    train_img, train_lbl = mnist.get_train_subset(0, 60_000)
    test_img, test_lbl = mnist.get_test_subset(0, 10000)

    if model_type == "numpy":
        return np.array(train_img), np.array(train_lbl), np.array(test_img), np.array(test_lbl)
    else: 
        return train_img, train_lbl, test_img, test_lbl

def get_model(model_type, shape, activations):
    if model_type == "numpy":
        return NumpyMLP(shape, activations)
    elif model_type == "raw":
        return MLP(shape, activations)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="numpy", choices=["numpy", "raw"])
    args = parser.parse_args()

    # Hyperparameters
    shape = [784, 128, 10]
    activations = ["relu", "linear"]

    epochs = 3
    batch_size = 32
    learning_rate = 0.3
    
    mlp = get_model(args.model, shape, activations)

    train_img, train_lbl, test_img, test_lbl = load_data(args.model)

    # Train model        
    mlp.train(train_img, train_lbl, epochs, batch_size, learning_rate)

    # Evaluate trained model
    print(f"success rate: {mlp.evaluate(test_img, test_lbl)}")

if __name__ == "__main__":
    main()
