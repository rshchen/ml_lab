import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return (1/(1+np.exp(-x)))

def relu(x):
    return np.maximum(x, 0)

if __name__ == "__main__":
    # step function
    # X = np.linspace(-5,5,1000)
    # Y = step_function(X)
    # plt.plot(X,Y)
    # plt.xlim(-6,6)
    # plt.ylim(-0.1,1.1)
    # plt.title("Step Function")
    # plt.show()
    # sigmoid function
    # X = np.linspace(-5,5,1000)
    # Y = sigmoid(X)
    # plt.plot(X,Y)
    # plt.xlim(-6,6)
    # plt.ylim(-0.1,1.1)
    # plt.title("Sigmoid Function")
    # plt.show()
    # ReLU function
    # X = np.linspace(-5,5,1000)
    # Y = relu(X)
    # plt.plot(X,Y)
    # plt.xlim(-6,6)
    # plt.ylim(-0.1,5)
    # plt.title("ReLU Function")
    # plt.show()
    # 宣告 1 列 3 行的畫布
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # 共用同一個橫軸輸入數據
    x = np.linspace(-5, 5, 1000)

    # 1. Step Function (調整獨立的 ylim 讓曲線更清晰)
    ax[0].plot(x, step_function(x), color='tab:blue', linewidth=2)
    ax[0].set_xlim(-6, 6)
    ax[0].set_ylim(-0.1, 1.1)
    ax[0].set_title("Step Function")
    ax[0].grid(True, linestyle=':', alpha=0.6)

    # 2. Sigmoid Function (調整獨立的 ylim 觀察平滑曲線)
    ax[1].plot(x, sigmoid(x), color='tab:orange', linewidth=2)
    ax[1].set_xlim(-6, 6)
    ax[1].set_ylim(-0.1, 1.1)
    ax[1].set_title("Sigmoid Function")
    ax[1].grid(True, linestyle=':', alpha=0.6)

    # 3. ReLU Function (縱軸拉高至 5，呈現非對稱線性特徵)
    ax[2].plot(x, relu(x), color='tab:green', linewidth=2)
    ax[2].set_xlim(-6, 6)
    ax[2].set_ylim(-0.1, 5.0)
    ax[2].set_title("ReLU Function")
    ax[2].grid(True, linestyle=':', alpha=0.6)

    # 自動調整子圖間距，避免標籤重疊
    plt.tight_layout()
    plt.show()