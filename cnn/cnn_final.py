import numpy as np


class Conv2d:
    def __init__(self, cin, cout, ksize, stride):
        self.cin = cin
        self.cout = cout
        self.ksize = ksize
        self.stride = stride

        self.w = np.random.rand(cout, cin, ksize, ksize)
        self.b = np.random.rand(cout)

    def forward(self, x):
        batch, h, w, _ = x.shape

        out_w = (h - self.ksize) // self.stride + 1
        out_h = (w - self.ksize) // self.stride + 1

        # cols shape (batch, out_w * out_h, cin*ksize*ksize)
        self.cols = self._im2col(x)

        # flat shape (cout, cin*ksize*ksize)
        flat_w = self.w.reshape(self.cout, -1)

        self.z = self.cols @ flat_w.T + self.b

        return self.z.reshape(batch, out_h, out_w, self.cout)

    # Convert input patch to colum matrix
    def _im2col(self, x):
        assert x.ndim == 4
        batch, h, w, cin = x.shape

        out_w = (w - self.ksize) // self.stride + 1
        out_h = (h - self.ksize) // self.stride + 1

        out = np.zeros((batch, out_w * out_h, cin * self.ksize * self.ksize))
        for i, row in enumerate(range(0, h - self.ksize + 1, self.stride)):
            for j, col in enumerate(range(0, w - self.ksize + 1, self.stride)):
                patch = x[:, row : row + self.ksize, col : col + self.ksize, :]
                out[:, i * row + j, :] = patch.reshape(batch, -1)

        return out
    
    def backward(self, grad):
        # grad shape (batch, out_h, out_w, cout)
        batch, out_h, out_w, _ = grad.shape

        # grad_flat shape (batch, out_h * out_w, cout)
        grad_flat = grad.reshape(batch, -1, self.cout)

        # self.cols shape (batch, out_h * out_w, cin*ksize*ksize)

        self.dW = grad * self.x

        # MAYBE HERE DON'T USE INPUT BUT IM2COL ???


conv = Conv2d(3, 2, 3, 1)

x = np.full((8, 8, 3), 2.0)
x = x[np.newaxis, ...]
a = np.repeat(x, 2, axis=0)

print(a.shape)

# res = conv.forward(x)
# print(res.shape)
