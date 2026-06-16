import numpy as np
from ml_lab.gradient import numerical_gradient_general
from ml_lab.layers import AffineLayer

class SimpleLossLayer:
    """
    用比較簡單的函數模擬 loss 函數
    核心映射：L = Y.sum(axis=1).mean()
    """
    def __init__(self):
        self.y_shape = None

    def forward(self, y):
        self.y_shape = y.shape
        return y.sum(axis=1).mean()
    
    def backward(self, dout=1.0):
        # 利用鏈鎖律計算每個格子的變化率都是 1 / 樣本數 M
        # 樣本數就是 y_shape[0]
        M = self.y_shape[0]
        dy = np.ones(self.y_shape) / M
        return dy * dout




def target_loss_function(affine_out):
    """
    用簡單加法與批次取平均模擬損失函數
    """
    return affine_out.sum(axis=1).mean()

if __name__ == "__main__":
    # 建立節點
    affine_node = AffineLayer()
    loss_node = SimpleLossLayer()
    # 宣告初始值 (3, 4)
    X_fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # (2, 3)
    init_w = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ])
    init_b = np.array([[0.1, 0.1, 0.1, 0.1]])
    lr = 0.1 # 學習速率
    # 數值梯度下降法
    w_num = init_w.copy()
    b_num = init_b.copy()
    # 梯度下降
    for _ in range(10):
        w_num_grad = numerical_gradient_general(lambda w: target_loss_function(affine_node.forward(X_fixed, w, b_num)), w_num)
        b_num_grad = numerical_gradient_general(lambda b: target_loss_function(affine_node.forward(X_fixed, w_num, b)), b_num)
        w_num -= lr * w_num_grad
        b_num -= lr * b_num_grad
    print(f"數值梯度下降法後的：\n權重\n{w_num}\n偏置\n{b_num}")
    
    # 自動微分梯度下降法
    w_auto = init_w.copy()
    b_auto = init_b.copy()

    for _ in range(10):
        # 先正向傳播
        affine_out = affine_node.forward(X_fixed, w_auto, b_auto)
        loss_out = loss_node.forward(affine_out)
        # 反向求梯度
        d_affine_out = loss_node.backward()
        dx, dw, db = affine_node.backward(d_affine_out)
        # 梯度下降
        w_auto -= lr * dw
        b_auto -= lr * db
    print(f"自動微分梯度下降法後的：\n權重\n{w_auto}\n偏置\n{b_auto}")



        
