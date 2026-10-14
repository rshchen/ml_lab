import numpy as np
# 透過本地套件專案管理進行優雅引用
from ml_lab.activations import relu, identity_function

def three_layer_net_forward(X, W1, b1, W2, b2, W3, b3):
    """
    通用多層感知機向前傳播函數
    """
    # --- 第 1 層 (隱藏層 1) ---
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    
    # --- 第 2 層 (隱藏層 2) ---
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    
    # --- 第 3 層 (輸出層) ---
    Z3 = np.dot(A2, W3) + b3
    Y_pred = identity_function(Z3)
    
    return Y_pred, A1, A2

if __name__ == "__main__":
    # --- 1. 親手建立輸入矩陣 (1 筆樣本，2 個特徵) ---
    X = np.array([[1.0, 0.5]])

    # --- 2. 親手根據連線看圖刻出權重矩陣 ---
    # 形狀必為 (2, 3)，對齊 (前層 2, 本層 3)
    W1 = np.array([
        [0.1, 0.3, 0.5],  # x1 連向後方 3 個神經元的權重
        [0.2, 0.4, 0.6]   # x2 連向後方 3 個神經元的權重
    ])
    b1 = np.array([0.1, 0.2, 0.3])
    
    # 形狀必為 (3, 2)，對齊 (前層 3, 本層 2)
    W2 = np.array([
        [0.1, 0.4],  # 隱藏層1的第1個節點連向隱藏層2的2個權重
        [0.2, 0.5],  # 隱藏層1的第2個節點連向隱藏層2的2個權重
        [0.3, 0.6]   # 隱藏層1的第3個節點連向隱藏層2的2個權重
    ])
    b2 = np.array([0.1, 0.2])
    
    # 形狀必為 (2, 2)，對齊 (前層 2, 本層 2)
    W3 = np.array([
        [0.1, 0.3],  # 隱藏層2的2個節點連向輸出 y1 的權重
        [0.2, 0.4]   # 隱藏層2的2個節點連向輸出 y2 的權重
    ])
    b3 = np.array([0.1, 0.2])

    # --- 3. 執行向前傳播 ---
    try:
        Y_pred, hidden1, hidden2 = three_layer_net_forward(X, W1, b1, W2, b2, W3, b3)
        
        # --- 4. 驗證結果 ---
        print("恭喜！看圖編排矩陣完全成功，維度完美對齊！")
        print("-" * 40)
        print("輸入 (X) 形狀:", X.shape)
        print("隱藏層 1 (A1) 形狀:", hidden1.shape)  # 預期 (1, 3)
        print("隱藏層 2 (A2) 形狀:", hidden2.shape)  # 預期 (1, 2)
        print("最終輸出 (Y_pred) 形狀:", Y_pred.shape)  # 預期 (1, 2)
        print("-" * 40)
        print("最終預測數值 Y_pred:\n", Y_pred)
        
    except ValueError as e:
        print("矩陣維度對齊失敗！請重新核對你的 W 與 b 的形狀設定：")
        print(e)

