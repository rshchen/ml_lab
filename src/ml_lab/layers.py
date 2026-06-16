import numpy as np

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