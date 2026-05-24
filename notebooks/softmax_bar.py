import numpy as np
import matplotlib.pyplot as plt

def softmax(X):
    if X.ndim == 2:
        exp_X = np.exp(X-X.max(axis=1, keepdims=True))
        result = exp_X/exp_X.sum(axis=1, keepdims=True) 
        return result
    else:
        exp_X = np.exp(X - X.max())
        result = exp_X/exp_X.sum()
        return result

    

if __name__ == "__main__":
    # X = np.array([[1,2],[2,9]])
    # print(softmax(X))
    # X = np.array([1,2,3])
    # print(softmax(X))
    categories = ['Cat', 'Dog', 'Rabbit']
    logits = np.array([2.0, 1.0, 0.1])

    probabilities = softmax(logits)
    
    fig, ax = plt.subplots(1, 2)

    ax[0].bar(categories, logits, color='tab:blue', alpha=0.8)
    ax[0].set_title("Original Scores (Logits)")
    ax[0].set_xlabel("Categories")
    ax[0].set_ylabel("Scores")
    ax[0].grid(axis='y', linestyle=':', alpha=0.6)

    ax[1].bar(categories, probabilities, color='tab:orange', alpha=0.8)
    ax[1].set_title("Softmax Probabilities")
    ax[1].set_xlabel("Categories")
    ax[1].set_ylabel("Probability")
    ax[1].set_ylim(0, 1.1)  # 固定縱軸範圍在 0 到 1.1，完美呈現機率邊界
    ax[1].grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()



