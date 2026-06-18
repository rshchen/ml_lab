import numpy as np
from ml_lab.activations import sigmoid, relu

class AffineLayer:
    def __init__(self):
        self.x = None
        self.w = None
        self.b = None
    def forward(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b
        out = x @ w + b
        return out 
    def backward(self, dout):
        dx = dout @ self.w.T
        dw = self.x.T @ dout
        db = dout.sum(axis=0, keepdims=True)
        return dx, dw, db
    
class ReluLayer:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x<=0)
        return relu(x)
    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx
    
class SigmoidLayer:
    def __init__(self):
        self.y = None
    def forward(self, x):
        out = self.y = sigmoid(x)
        return out
    def backward(self, dout):
        y = self.y
        dx = dout * y * (1 - y)
        return dx
    


