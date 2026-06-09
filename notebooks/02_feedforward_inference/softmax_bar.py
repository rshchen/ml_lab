import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    if x.ndim == 2:
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))
        result = exp_x / exp_x.sum(axis=1, keepdims=True) 
        return result
    else:
        exp_x = np.exp(x - x.max())
        result = exp_x / exp_x.sum()
        return result

    

if __name__ == "__main__":
    # x = np.array([[1,2],[2,9]])
    # print(softmax(x))
    # x = np.array([1,2,3])
    # print(softmax(x))
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



