import numpy as np
from ml_lab.gradient import numerical_gradient_general
from ml_lab.layers import AffineLayer, SoftmaxWithCeeLayer

def classification_black_box(x_fixed, w_input, b_input, t_fixed, affine_node, softmax_loss_node):
    out_affine = affine_node.forward(x_fixed, w_input, b_input)
    loss = softmax_loss_node.forward(out_affine, t_fixed)
    return loss

if __name__ == "__main__":
    affine_node = AffineLayer()
    softmax_loss_node = SoftmaxWithCeeLayer()
    
    X_fixed = np.array([[0.5, 1.2], [2.1, -0.8]])  # (2, 2)
    T_fixed = np.array([[1, 0, 0], [0, 0, 1]])     # (2, 3) One-Hot 標籤
    
    init_w = np.array([
        [0.2, -0.1, 0.4],
        [0.3, 0.5, -0.2]
    ])                                             # (2, 3)
    init_b = np.array([[0.0, 0.0, 0.0]])           # (1, 3)
    lr = 0.2
    
    # --- 軌跡 A：數值梯度下降 ---
    w_num = init_w.copy()
    b_num = init_b.copy()
    for _ in range(10):
        w_num_grad = numerical_gradient_general(
            lambda w: classification_black_box(X_fixed, w, b_num, T_fixed, affine_node, softmax_loss_node), w_num
        )
        b_num_grad = numerical_gradient_general(
            lambda b: classification_black_box(X_fixed, w_num, b, T_fixed, affine_node, softmax_loss_node), b_num
        )
        w_num -= lr * w_num_grad
        b_num -= lr * b_num_grad

    # --- 軌跡 B：自動微分（計算圖級聯） ---
    w_auto = init_w.copy()
    b_auto = init_b.copy()
    for _ in range(10):
        out_affine = affine_node.forward(X_fixed, w_auto, b_auto)
        loss_val = softmax_loss_node.forward(out_affine, T_fixed)
        
        d_affine_out = softmax_loss_node.backward(1.0)
        dx, dw, db = affine_node.backward(d_affine_out)
        
        w_auto -= lr * dw
        b_auto -= lr * db

    print(f"=== 聯合拓撲 10 步軌跡對帳 ===")
    print(f"W 權重軌跡絕對誤差: {np.mean(np.abs(w_num - w_auto))}")
    print(f"b 偏置軌跡絕對誤差: {np.mean(np.abs(b_num - b_auto))}")

