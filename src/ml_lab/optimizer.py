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
        x = x - lr * grad
        
    return x