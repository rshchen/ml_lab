import numpy as np
from ml_lab.layers import Convolution


def numerical_gradient_array(f, x, h=1e-5):
    """計算張量 f(x) 關於 x 的數值梯度"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
        it.iternext()
        
    return grad


def test_convolution_gradient_check():
    """使用數值梯度驗證無狀態 Convolution 層的反向傳播"""
    np.random.seed(42)

    x = np.random.randn(2, 2, 4, 4)
    w = np.random.randn(2, 2, 3, 3)
    b = np.random.randn(2)

    conv = Convolution(stride=1, padding=1)

    # 解析梯度 (Analytic Gradient) - 無狀態算子風格
    out = conv.forward(x, w, b)
    dout = np.random.randn(*out.shape)
    dx_analytic, dw_analytic, db_analytic = conv.backward(dout)

    # 數值梯度 (Numerical Gradient)
    f_x = lambda x_val: np.sum(conv.forward(x_val, w, b) * dout)
    f_w = lambda w_val: np.sum(Convolution(stride=1, padding=1).forward(x, w_val, b) * dout)
    f_b = lambda b_val: np.sum(Convolution(stride=1, padding=1).forward(x, w, b_val) * dout)

    dx_numerical = numerical_gradient_array(f_x, x)
    dw_numerical = numerical_gradient_array(f_w, w)
    db_numerical = numerical_gradient_array(f_b, b)

    # 計算相對誤差 (Relative Error)
    rel_error_dx = np.max(np.abs(dx_analytic - dx_numerical) / (np.abs(dx_analytic) + np.abs(dx_numerical) + 1e-8))
    rel_error_dw = np.max(np.abs(dw_analytic - dw_numerical) / (np.abs(dw_analytic) + np.abs(dw_numerical) + 1e-8))
    rel_error_db = np.max(np.abs(db_analytic - db_numerical) / (np.abs(db_analytic) + np.abs(db_numerical) + 1e-8))

    print(f"dx 相對誤差: {rel_error_dx:.2e}")
    print(f"dW 相對誤差: {rel_error_dw:.2e}")
    print(f"db 相對誤差: {rel_error_db:.2e}")

    assert rel_error_dx < 1e-4, "dx 梯度檢查失敗！"
    assert rel_error_dw < 1e-4, "dW 梯度檢查失敗！"
    assert rel_error_db < 1e-4, "db 梯度檢查失敗！"
    print("無狀態卷積層反向傳播數值梯度檢查全數通過！")


if __name__ == "__main__":
    test_convolution_gradient_check()

