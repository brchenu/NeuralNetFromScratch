import random, math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivate(x):
    return sigmoid(x) * (1 - sigmoid(x))

def loss_function(truth, pred):
    assert(len(truth) == len(pred))

    errors = [(t - p)**2 for t, p in zip(truth, pred)]
    return sum(errors) / 2*len(truth) # Divide by two for derivate convinence

class Neuron:
    def __init__(self, ninputs):
        self.bias = random.random()
        self.weights = [random.random() for _ in range(ninputs)]

    # Forward Pass
    # z = w*x + b
    # a = f(z) where f is the activation function
    def forward(self, inputs):
        self.z = self.bias
        self.last_inputs = inputs
        for x, w in zip(inputs, self.weights): 
            self.z += x * w

        self.y = sigmoid(self.z)
        return self.y

    def backward(self, grad_out):
        # ∂L/∂a = grad_out

        # Get Derivate of L w.r.t z, i.e: ∂L/∂z
        # Chaine rule apply: ∂L/∂z = ∂L/∂a * ∂a/∂z
        #
        # where ∂a/∂z is equal to the derivate of the actiation function
        grad_z = grad_out * sigmoid_derivate(self.z)
        
        # Get derivate of L w.r.t to w, i.e: ∂L/∂w
        # Chaine rule apply: ∂L/∂w = ∂L/∂a * ∂a/∂z * ∂z/∂w 
        #                or: ∂L/∂w =     ∂L/∂z     * ∂z/∂w

        # ∂z/∂w is the derivate of: x*w + b w.r.t w which is equal to x
        # ∂a/∂z is the previously calculated derivate which gives us:

        # Note the this gradient will then be used to update the bias to minimize the loss
        self.grad_w = [x * grad_z for x in self.last_inputs] 

        # Get Derivate of L w.r.t b, i.e: ∂L/∂b
        # Chaine rule apply: ∂L/∂b = ∂L/∂a * ∂a/∂z * ∂z/∂b
        #                or: ∂L/∂b =      ∂L/∂z    * ∂z/∂b

        # ∂z/∂b is the derivate of: x*w + b, w.r.t b which is equal to 1
        # So we got: ∂L/∂b = ∂L/∂z * 1 = ∂L/∂z

        # Note this gradient will then be used to update the bias to minimize the loss
        self.grad_b = grad_z

        # Finally we want to propagate the gradient to the next layer
        # We want the derviate of L w.r.t to x, i.e: ∂L/∂x
        
        # ∂L/∂x = ∂L/∂a * ∂a/∂z * ∂z/∂x
        # ∂L/∂x =     ∂L/∂z     * ∂z/∂x

        # ∂z/∂x is the derivate of: x*w + b, w.r.t x which is equal to w
        grad_x = [w * grad_z for w in self.weights]

        return grad_x
    
    def update(self, learning_rate):
        # Update weights
        for grad, w in zip(self.grad_w, self.weights):
            w = w - grad * learning_rate
        self.bias = self.bias - self.grad_b * learning_rate

class Layer:
    def __init__(self, ninputs, nneurons):
        self.neurons = [Neuron(ninputs) for _ in range(nneurons)]

    def forward(self, inputs):
        outputs = []
        for n in self.neurons:
            outputs.append(n.forward(inputs))

        return outputs
    
    def backward(self, grad):
        assert(len(grad) == len(self.neurons)) 
        
        # We need to sum the gradients returned by our different neurons for the same inputs
        # because one inputs affect the N neurons of our layers so we need to sum their contributions
        #
        # Ex: If I have 2 Neurons, and 3 inputs (x1, x2, x3), each neurons backward will return [x1_grad, x2_grad, x3_grad]
        # So we need to sum the x1_grad of the neuron N1 with the x1_grad of the neuron N2 in order to have
        # the entire impact of x1 before propagating it back to the previous layer.
        #
        # The reason we do that as mentionned above is because if we want to know the impact of x1 on the Loss 
        # we need to sum up all of its contribution.

        summed_out = None
        for g, n in zip(grad, self.neurons):
            out = n.backward(g)
            if summed_out is None:
                summed_out = out[:]
            else:
                summed_out = [t + o for t, o in zip(summed_out, out)]

        return summed_out
    
    def update(self, learning_rate):
        for n in self.neurons:
            n.update(learning_rate)

class MLP:
    def __init__(self, layers_params: int):
        self.layers = []
        for l in layers_params:
            assert(len(l) == 2)
            self.layers.append(Layer(l[0], l[1]))

    def forward(self, inputs):
        current_in = inputs
        for l in self.layers:
            current_in = l.forward(current_in)

        return current_in
    
    def update_params(self, learning_rate):
        for l in self.layers:
            l.update(learning_rate)

    def train(self, inputs, truth):
        pred = self.forward(inputs)
        loss = loss_function(truth, pred)

        print(f"loss:{loss}")

        # Here we do: pred - truth, because its the derivative of our loss_function
        grad_loss = [p - t for p,t in zip(pred, truth)]

        grad = grad_loss
        for l in reversed(self.layers):
            grad = l.backward(grad)

if __name__ == "__main__":

    mlp = MLP([[1, 3], [5, 2]])

    truth = [2, 4]
    x = [8]

    pred = mlp.forward(x)
    loss = loss_function(truth, pred)