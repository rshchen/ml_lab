import numpy as np
from ml_lab.activations import sigmoid, relu, softmax
from ml_lab.losses import cross_entropy_error

class AffineLayer:
    def __init__(self):
        self.x = None
        self.w = None
        self.b = None
    def forward(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b
        out = x @ w + b
        return out 
    def backward(self, dout):
        dx = dout @ self.w.T
        dw = self.x.T @ dout
        db = dout.sum(axis=0, keepdims=True)
        return dx, dw, db
    
class ReluLayer:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x<=0)
        return relu(x)
    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx
    
class SigmoidLayer:
    def __init__(self):
        self.y = None
    def forward(self, x):
        out = self.y = sigmoid(x)
        return out
    def backward(self, dout):
        y = self.y
        dx = dout * y * (1 - y)
        return dx
    
class MeanSquaredErrorLayer:
    """
    均方誤差 (Mean Squared Error) - 專治【回歸問題】
    支援單筆 1D 向量與 Batch 2D 矩陣
    
    物理維度與格式限制：
    1. 本函數「不支援」未轉換的分類數字標籤 (如 [2, 0, 4])。
    2. 真實標籤 t 必須是與預測資料 y 格式完全一致的「浮點數矩陣」或「One-Hot 矩陣」。
    3. 在回歸任務中，t 的數值代表真實的實數連續目標（如房價、坐標）。

    為了不重複升維防線，因此這邊沒有直接 from ml_lab.losses import mean_squared_error
    """
    def __init__(self):
        self.y = None
        self.t = None
    def forward(self, y, t):
        # 升維防線
        if y.ndim == 1:
            y = y.reshape(1, -1)
            t = t.reshape(1, -1)
        self.y = y
        self.t = t
        sample_losses = 0.5 * np.sum((y - t) ** 2, axis=1)

        out = np.mean(sample_losses)
        return out
    def backward(self, dout):
        # 此時的 self.y 必定是 (Batch, Feature) 的 2D 矩陣
        batch_size = self.y.shape[0]
        dy = (self.y - self.t)/batch_size
        return dy * dout

class SoftmaxWithCeeLayer:
    def __init__(self):
        self.y = None
        self.t = None
    def forward(self, x, t):
        self.t = t
        self.y = y = softmax(x)
        loss = cross_entropy_error(y, t)
        return loss
    def backward(self, dout=1.0):
        y = self.y
        t = self.t
        batch_size = 1 if y.ndim == 1 else y.shape[0]
        
        # 只要不是 One-Hot (尺寸不相等)，就代表是傳統數字標籤
        if t.size != y.size:
            t_one_hot = np.zeros_like(y)
            # 當 y 是 1D (單筆) 時，batch_size=1，t_one_hot[t] 直接利用 1D 索引變為 1
            # 當 y 是 2D (批次) 時，利用 np.arange 進行二維進階索引
            if y.ndim == 1:
                t_one_hot[t] = 1
            else:
                t_one_hot[np.arange(batch_size), t] = 1
            t = t_one_hot
            
        dx = (y - t) / batch_size
        return dx * dout





        

