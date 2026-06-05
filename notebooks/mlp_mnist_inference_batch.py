import pickle
import numpy as np
from sklearn.datasets import fetch_openml
from ml_lab.activations import sigmoid, softmax

def init_network():
    with open("data/sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    """
    支援批次（Batch）運算的高階向前傳播函數 (依據先前探針得到的 W1(784,50), W2(50,100), W3(100,10) 維度對齊)
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
    network = init_network()
    
    print("正在載入 1000 筆 MNIST 測試數據...")
    mnist = fetch_openml("mnist_784", version=1, data_home="data/", as_frame=True)
    
    # 提取前 1000 筆完整數據
    X_batch = mnist.data.head(1000).to_numpy()
    y_batch = mnist.target.head(1000).to_numpy()
    
    # 將 1000 筆樣本的像素進行標準正規化
    X_batch_norm = X_batch / 255
    
    # 執行全量批次推理，probabilities 預期形狀為 (1000, 10)
    probabilities = predict(network, X_batch_norm)
    
    # 利用正確的 axis 參數，橫向抓出每筆樣本機率最高的數字
    pred_labels = np.argmax(probabilities, axis=1)
    
    # 將 y_batch 轉換為整數型態，以便與 pred_labels 進行直接比對
    y_batch_int = y_batch.astype(int)
    
    # 4. 請填空：計算這 1000 筆裡面，有幾筆預測正確
    correct_count = np.sum(pred_labels==y_batch_int)
    
    # 計算準確率 (Accuracy)
    accuracy = correct_count / len(y_batch_int)
    
    print("\n========================================")
    print(" 1000 筆 MNIST 批次向前傳播推理報告")
    print("========================================")
    print(f"輸入矩陣 X_batch 形狀: {X_batch.shape}")
    print(f"輸出機率矩陣形狀     : {probabilities.shape}")
    print(f"預測正確的總樣本數   : {correct_count} 筆")
    print(f"這套權重的最終準確率 : {accuracy:.2%}")
    print("========================================")

