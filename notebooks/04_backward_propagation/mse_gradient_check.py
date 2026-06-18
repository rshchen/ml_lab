import numpy as np
from ml_lab.gradient import numerical_gradient_general
from ml_lab.layers import AffineLayer, SigmoidLayer, MeanSquaredErrorLayer

def black_box(x_fixed, w_input, b_input, t_fixed, affine_node, sigmoid_node, mse_node):
    out_affine = affine_node.forward(x_fixed, w_input, b_input)
    out_sigmoid = sigmoid_node.forward(out_affine)
    loss = mse_node.forward(out_sigmoid, t_fixed)
    return loss

if __name__ == "__main__":
    # 建立節點
    affine_node = AffineLayer()
    sigmoid_node = SigmoidLayer()
    mse_node = MeanSquaredErrorLayer()
    
    # 宣告初始值與真實標籤 T
    X_fixed = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # (2, 3)
    T_fixed = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]) # (2, 4)
    
    init_w = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ]) # (3, 4)
    init_b = np.array([[0.1, 0.1, 0.1, 0.1]]) # (1, 4)
    lr = 0.5
    
    # --- 數值梯度下降軌跡 ---
    w_num = init_w.copy()
    b_num = init_b.copy()
    for _ in range(10):
        w_num_grad = numerical_gradient_general(
            lambda w: black_box(X_fixed, w, b_num, T_fixed, affine_node, sigmoid_node, mse_node), w_num
        )
        b_num_grad = numerical_gradient_general(
            lambda b: black_box(X_fixed, w_num, b, T_fixed, affine_node, sigmoid_node, mse_node), b_num
        )
        w_num -= lr * w_num_grad
        b_num -= lr * b_num_grad

    # --- 自動微分梯度下降軌跡 ---
    w_auto = init_w.copy()
    b_auto = init_b.copy()
    for _ in range(10):
        # 正向傳播記憶狀態
        out_affine = affine_node.forward(X_fixed, w_auto, b_auto)
        out_sigmoid = sigmoid_node.forward(out_affine)
        loss_val = mse_node.forward(out_sigmoid, T_fixed)
        
        # 反向連鎖火炬接力
        d_mse = mse_node.backward(1.0)
        d_sigmoid = sigmoid_node.backward(d_mse)
        dx, dw, db = affine_node.backward(d_sigmoid)
        
        w_auto -= lr * dw
        b_auto -= lr * db

    # 誤差校對
    w_error = np.mean(np.abs(w_num - w_auto))
    b_error = np.mean(np.abs(b_num - b_auto))
    print(f"10 步軌跡後 W 矩陣絕對誤差: {w_error}")
    print(f"10 步軌跡後 b 矩陣絕對誤差: {b_error}")

