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
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay
        
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
            
            # 僅對權重 W 施加衰減
            decay_term = self.weight_decay * params[key] if 'W' in key else 0.0
            grad_total = grads[key] + decay_term
            
            # 依據 SGD 代數公式實作參數原地更新
            params[key] -= self.lr * grad_total


class Momentum:
    """動量優化器 (Momentum Optimization)"""
    def __init__(self, lr=0.01, decay=0.9, weight_decay=0.0):
        self.lr = lr
        self.decay = decay  # 動量衰減/阻力係數
        self.weight_decay = weight_decay
        self.v = {}         # 儲存全域參數速度的字典
        
    def update(self, params, grads):
        if not self.v:
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            decay_term = self.weight_decay * params[key] if 'W' in key else 0.0
            grad_total = grads[key] + decay_term
            # 1. 舊速度打折（因摩擦力衰減）並疊加當前包含 Weight Decay 的負梯度加速度
            self.v[key] = self.decay * self.v[key] - self.lr * grad_total
            # 2. 依據更新後的速度更新參數位置
            params[key] += self.v[key]

class AdaGrad:
    """自適應學習率優化器 (Adaptive Gradient)"""
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay
        self.h = {}  # 儲存全域參數梯度平方和的字典 (形狀與參數完全守恆)
        
    def update(self, params, grads):
        """
        params, grads: 與 SGD/Momentum 介面完全相同的參數與梯度字典
        """
        # 如果累積字典目前為空，說明是第一步前進，必須發動動態延遲初始化
        if not self.h:
            for key, val in params.items():
                # ------------------------------------------------------------
                # 根據參數的幾何形狀，初始化一模一樣的全零累積張量
                # ------------------------------------------------------------
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            # ------------------------------------------------------------
            # 依據 AdaGrad 累積與參數更新公式實作參數原地改命
            # 1. 點對點累加梯度的平方到 self.h[key]
            # 2. 計算自適應調整後的步長，並原地更新 params[key] (加上 1e-7 防止除以零)
            # ------------------------------------------------------------
            decay_term = self.weight_decay * params[key] if 'W' in key else 0.0
            grad_total = grads[key] + decay_term

            # 1. 點對點累加總梯度的平方到 self.h[key]
            self.h[key] += np.square(grad_total)
            # 2. 計算自適應調整後的步長，並原地更新 params[key]
            params[key] -= self.lr * grad_total / (np.sqrt(self.h[key]) + 1e-7)

class Adam:
    """終極完全體優化器 (Adam)"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.iter = 0  # 追蹤當前的迭代步數 t
        self.m = {}    # 一階矩緩衝字典 (速度)
        self.v = {}    # 二階矩緩衝字典 (憤怒值)
        
    def update(self, params, grads):
        if not self.m:
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                
        # 步數計數器累加，用於偏差修正的指數計算
        self.iter += 1
        
        # 計算偏差修正的係數項（已做代數合併）
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for key in params.keys():
            decay_term = self.weight_decay * params[key] if 'W' in key else 0.0
            grad_total = grads[key] + decay_term
            
            # 依據 Adam 公式更新一階矩 self.m 與二階矩 self.v
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad_total
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * np.square(grad_total)
            
            # 結合偏差修正項 lr_t，實作參數原地更新
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)