import numpy as np
from ml_lab.activations import softmax
from ml_lab.losses import cross_entropy_error
from ml_lab.gradient import numerical_gradient_general
from ml_lab.optimizer import gradient_descent

class SimpleNet:
    """單層權重神經網路模型"""
    def __init__(self):
        # 初始化一個 2x3 的標準常態分佈權重矩陣 W
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        """前向傳播：計算分數 (Scores/Logits)"""
        # 實作輸入向量與權重矩陣的仿射變換 (Affine Transformation)
        return x @ self.W 

    def loss(self, x, t):
        """複合目標函數：計算當前參數下的 Cross Entropy Error"""
        z = self.predict(x)
        y = softmax(z)
        loss_val = cross_entropy_error(y, t)
        return loss_val

if __name__ == "__main__":
    net = SimpleNet()
    # 固定的輸入資料 x (常數) 與期望的 One-hot 分類標籤 t (常數)
    x_data = np.array([0.6, 0.9])
    t_label = np.array([0, 0, 1]) # 期望正確分類為第三類
    
    print(f"初始隨機權重矩陣 W:\n{net.W}\n")
    print(f"預測的分數{net.predict(x_data)}")
    # 建立一個只接收 W 為自變數的純粹黑盒子函數 f
    def f_dummy(W):
        net.W = W
        return net.loss(x_data, t_label)
    
    # 調用通用多維數值微分算子，對權重矩陣直接求取偏導數矩陣 dW
    dW = numerical_gradient_general(f_dummy, net.W)
    
    print(f"數值微分計算出的權重梯度矩陣 dW (形狀必須與 W 完全一致):\n{dW}\n")
    
    # 模擬一步手動梯度下降更新 (假設學習率 lr = 0.1)
    # 實作離散更新公式：W = W - lr * dW
    lr = 0.1
    net.W -= lr * dW 
    print(f"執行一步 GD 優化後的新權重矩陣 W:\n{net.W}")
    print(f"預測的分數{net.predict(x_data)}")

    # 模擬多步梯度下降
    net.W = gradient_descent(f_dummy, net.W)
    print(f"執行 100 步 GD 優化後的新權重矩陣 W:\n{net.W}")
    print(f"預測的分數{net.predict(x_data)}")

