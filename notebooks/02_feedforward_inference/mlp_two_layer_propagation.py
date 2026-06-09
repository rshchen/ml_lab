import numpy as np
from ml_lab.activations import relu, softmax

def two_layer_net_forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 +b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 +b3
    Y = softmax(Z3)
    return Y


if __name__=="__main__":
    X = np.array([1.0, 0.5])
    W1 = np.array(
        [[0.1, 0.3, 0.5],
         [0.2, 0.4, 0.6]])
    b1 = np.array([0.1, 0.2, 0.3])
    W2 = np.array(
        [[0.1, 0.4],
         [0.2, 0.5],
         [0.3, 0.6]])
    b2 = np.array([0.1, 0.2])
    W3 = np.array([
        [0.1, 0.3],
        [0.2, 0.4]
    ])
    b3 = np.array([0.1, 0.2])
    result = two_layer_net_forward(X, W1, b1, W2, b2, W3, b3)
    print(f"Answer is: {result}")