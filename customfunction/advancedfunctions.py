import numpy as np
from customfunction.functions import Function
from customfunction.variable import Variable

class Linear(Function):
    def forward(self, x, W, b):
        y = np.dot(x, W) + b
        return y
    
    def backward(self, gy):
        x, W, b = self.inputs[0].data, self.inputs[1].data, self.inputs[2].data
        gx = np.dot(gy, W.T)
        gW = np.dot(x.T, gy)
        gb = np.sum(gy, axis=0)
        return gx, gW, gb

def linear(x, W, b):
    return Linear()(x, W, b)


class ReLU(Function):
    def forward(self, x):
        y = np.maximum(0, x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * (x > 0)
        return gx

def relu(x):
    return ReLU()(x) 


class MeanSquaredError(Function):
    def forward(self, y, t):
        diff = y - t
        return np.mean(diff ** 2)
    
    def backward(self, gy):
        y, t = self.inputs[0].data, self.inputs[1].data
        n = len(y)
        gy = 2 * (y - t) / n
        return gy, -gy

def mean_squared_error(y, t):
    return MeanSquaredError()(y, t)
