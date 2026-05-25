import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def relu(x):
    return np.maximum(x, 0)

def identity_function(x):
    return x

def softmax(x):
    if x.ndim == 2:
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))
        result = exp_x / exp_x.sum(axis=1, keepdims=True) 
        return result
    else:
        exp_x = np.exp(x - x.max())
        result = exp_x / exp_x.sum()
        return result
