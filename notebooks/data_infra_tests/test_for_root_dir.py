import numpy as np
import matplotlib.pyplot as plt
from ml_lab import step_function, sigmoid, relu, softmax

if __name__ == "__main__":
    # step function
    X = np.linspace(-5,5,1000)
    Y = step_function(X)
    plt.plot(X,Y)
    plt.xlim(-6,6)
    plt.ylim(-0.1,1.1)
    plt.title("Step Function")
    plt.show()