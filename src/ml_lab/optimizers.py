import numpy as np
from ml_lab.gradient import numerical_gradient_general


def gradient_descent(f, init_x, lr=0.1, step_num=100):
    """梯度下降法核心引擎
    f: 目標損失函數 (黑盒子)
    init_x: 初始權重參數座標 (NumPy Array，支援任意維度)
    lr: 學習率 (η)
    step_num: 離散迭代步數
    """
    # [ 優化 ] 在建立副本的同時，強制將型態轉為 float64，一舉解決 copy 與整數截斷問題
    x = init_x.astype(np.float64, copy=True)
    
    for i in range(step_num):
        # 調用通用多維數值微分算子
        grad = numerical_gradient_general(f, x)
        
        # 實作離散參數更新公式 W^(k+1) = W^(k) - η * ∇f
        x -= lr * grad
        
    return x


class Sgd:
    """隨機梯度下降法 (Stochastic Gradient Descent)"""
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        """
        發動原地減法梯度更新
        params: 儲存網路權重與偏置的字典, 例如 {'W1': ..., 'b1': ...}
        grads: 儲存反向傳播梯度的一維/多維字典, 例如 {'W1': ..., 'b1': ...}
        """
        for key in params.keys():
            # ----------------------------------------------------------------
            # 依據 SGD 代數公式實作參數原地更新 (維持形狀守恆)
            # ----------------------------------------------------------------
            params[key] -= self.lr * grads[key]


class Momentum:
    """動量優化器 (Momentum Optimization)"""
    def __init__(self, lr=0.01, decay=0.9):
        self.lr = lr
        self.decay = decay  # 動量衰減/阻力係數
        self.v = {}         # 儲存全域參數速度的字典
        
    def update(self, params, grads):
        if not self.v:
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            # 1. 舊速度打折（因摩擦力衰減）並疊加當前負梯度加速度
            self.v[key] = self.decay * self.v[key] - self.lr * grads[key]
            # 2. 依據更新後的速度更新參數位置
            params[key] += self.v[key]