import math

# Loss Function
def meanSquareError(ground_truth: list, prediction: list):
    assert(len(ground_truth) == len(prediction))

    errors = [math.pow(a - b, 2) for a, b in zip(ground_truth, prediction)]
    return sum(errors) / len(ground_truth)

class Layer():
    weights = []
    inputs = []
    bias = 0
    value = 0

    def __init__(self, weights: list, inputs: list, bias: int):
        self.weights = weights
        self.inputs = inputs

    def sigmoid(self) -> float:
        return 1 / (1 + math.exp(-self.value))
    
    # repr should print a representation of the object 
    # (most likely one of the ways possible to create this object)
    def __repr__(self):
        return f"Layer(inputs={self.inputs})"