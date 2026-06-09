import numpy as np
def step_function(x):
    return np.where(x > 0, 1, 0)

def and_gate(X):
    W = np.array([[1], [1]])
    b = -1.5
    return step_function(X @ W + b)

def or_gate(X):
    W = np.array([[1], [1]])
    b = -0.5
    return step_function(X @ W + b)

def nand_gate(X):
    W = np.array([[-1], [-1]])
    b = 1.5
    return step_function(X @ W + b)


# p ^ q = (p | q) & (~(p&q))
def xor_gate(X):
    s1 = or_gate(X) # (p | q), shape = (4,1)
    s2 = nand_gate(X)  # (~(p & q)), shape = (4,1)
    x_layer2 = np.hstack((s1, s2))
    y = and_gate(x_layer2)
    return y

if __name__ == "__main__":
    X = np.array(
        [[0,0],
         [1,0],
         [0,1],
         [1,1]
        ])
    print(xor_gate(X))