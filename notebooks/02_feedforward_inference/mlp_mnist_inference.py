import pickle
import numpy as np
from sklearn.datasets import fetch_openml
from ml_lab.activations import sigmoid, softmax

def init_network():
    """
    開箱 pickle 罐頭，載入現成權重
    """
    with open("data/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    """
    三層 MLP 向前傳播推理函數 (依據先前探針得到的 W1(784,50), W2(50,100), W3(100,10) 維度對齊)
    """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    # 依照 Z = X @ W + b -> A = activation(Z) 的順序完成三層架構
    # 第一層運算 (784,) @ (784, 50) -> (50,)
    Z1 = x @ W1 + b1
    A1 = sigmoid(Z1)
    
    # 第二層運算 (50,) @ (50, 100) -> (100,)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    
    # 第三層 (輸出層) 運算 (100,) @ (100, 10) -> (10,)
    Z3 = A2 @ W3 + b3
    Y = softmax(Z3)
    
    return Y

if __name__ == "__main__":
    # 載入權重字典
    print("正在解鎖權重罐頭...")
    network = init_network()
    
    # 快速載入 1000 筆測試數據
    print("正在載入測試數據...")
    mnist = fetch_openml("mnist_784", version=1, data_home="data/", as_frame=True)
    X = mnist.data.head(1000)
    y = mnist.target.head(1000)
    
    # 提取第 0 筆影像
    sample_index = 0
    single_image_1d = X.iloc[sample_index].to_numpy()
    true_label = y.iloc[sample_index]
    
    # 核心正規化步驟，將像素縮放到 0.0 ~ 1.0
    x_norm = single_image_1d / 255
    
    # 執行預測
    probabilities = predict(network, x_norm)
    
    # 利用 numpy 函數找出最大機率對應的索引
    pred_label = np.argmax(probabilities)
    
    print("\n推理驗證結果：")
    print("-" * 30)
    print(f"網路預測的數字為: {pred_label}")
    print(f"圖片真實的答案為: {true_label}")
    print("-" * 30)
    
    if str(pred_label) == str(true_label):
        print("預測成功！神經網路完美識別出該手寫圖片！")
    else:
        print("預測失敗，請重新核對矩陣對齊與正規化步驟。")
