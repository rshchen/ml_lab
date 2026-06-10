import numpy as np
from ml_lab.optimizer import gradient_descent

def target_function_2d(x):
    """測試碗狀高維函數: f(x) = x00^2 + x01^2 + x10^2 + x11^2
    輸入 x 可以是任意形狀的 NumPy 陣列（例如 2x2 矩陣）
    """
    return np.sum(x**2)


if __name__ == "__main__":
    # 測試一個 2x2 的矩陣權重起點
    init_weight = np.array([[-3.0, 4.0], [2.0, -1.0]])
    
    # 執行通用數值梯度下降法
    final_weight = gradient_descent(target_function_2d, init_x=init_weight, lr=0.1, step_num=20)
    
    print(f"初始化矩陣權重起點:\n{init_weight}")
    print(f"經過 20 步迭代後的最終矩陣權重（必須收斂到 0 附近）:\n{final_weight}")

