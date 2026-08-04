import numpy as np
from ml_lab.layers import Pooling


def numerical_gradient_array(f, x, h=1e-5):
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


def test_max_pooling_forward_value():
    x = np.array([[
        [
            [1.0, 2.0, 8.0, 4.0],
            [5.0, 6.0, 7.0, 3.0],
            [9.0, 3.0, 2.0, 1.0],
            [4.0, 0.0, 5.0, 6.0]
        ]
    ]])

    pool = Pooling(pool_h=2, pool_w=2, stride=2, mode='max')
    out = pool.forward(x)

    expected_out = np.array([[
        [
            [6.0, 8.0],
            [9.0, 6.0]
        ]
    ]])

    assert np.allclose(out, expected_out), f"Max Pooling 前向傳播數值不符，得到:\n{out}"
    print("✓ Max Pooling 前向傳播數值測試通過！")


def test_avg_pooling_forward_value():
    x = np.array([[
        [
            [1.0, 2.0, 8.0, 4.0],
            [5.0, 6.0, 7.0, 3.0],
            [9.0, 3.0, 2.0, 1.0],
            [4.0, 0.0, 5.0, 6.0]
        ]
    ]])

    pool = Pooling(pool_h=2, pool_w=2, stride=2, mode='avg')
    out = pool.forward(x)

    expected_out = np.array([[
        [
            [3.5, 5.5],
            [4.0, 3.5]
        ]
    ]])

    assert np.allclose(out, expected_out), f"Avg Pooling 前向傳播數值不符，得到:\n{out}"
    print("✓ Avg Pooling 前向傳播數值測試通過！")


def test_pooling_gradient_check():
    np.random.seed(42)
    x = np.random.randn(2, 2, 4, 4)

    # Max Pooling 梯度檢查
    pool_max = Pooling(pool_h=2, pool_w=2, stride=2, mode='max')
    out_max = pool_max.forward(x)
    dout_max = np.random.randn(*out_max.shape)
    dx_analytic_max = pool_max.backward(dout_max)

    f_max = lambda x_val: np.sum(pool_max.forward(x_val) * dout_max)
    dx_numerical_max = numerical_gradient_array(f_max, x)

    rel_error_max = np.max(
        np.abs(dx_analytic_max - dx_numerical_max) / (np.abs(dx_analytic_max) + np.abs(dx_numerical_max) + 1e-8)
    )
    assert rel_error_max < 1e-4, f"Max Pooling 反向傳播梯度檢查失敗，相對誤差: {rel_error_max}"
    print(f"✓ Max Pooling 反向傳播梯度檢查通過，相對誤差: {rel_error_max:.2e}")

    # Avg Pooling 梯度檢查
    pool_avg = Pooling(pool_h=2, pool_w=2, stride=2, mode='avg')
    out_avg = pool_avg.forward(x)
    dout_avg = np.random.randn(*out_avg.shape)
    dx_analytic_avg = pool_avg.backward(dout_avg)

    f_avg = lambda x_val: np.sum(pool_avg.forward(x_val) * dout_avg)
    dx_numerical_avg = numerical_gradient_array(f_avg, x)

    rel_error_avg = np.max(
        np.abs(dx_analytic_avg - dx_numerical_avg) / (np.abs(dx_analytic_avg) + np.abs(dx_numerical_avg) + 1e-8)
    )
    assert rel_error_avg < 1e-4, f"Avg Pooling 反向傳播梯度檢查失敗，相對誤差: {rel_error_avg}"
    print(f"✓ Avg Pooling 反向傳播梯度檢查通過，相對誤差: {rel_error_avg:.2e}")


if __name__ == "__main__":
    test_max_pooling_forward_value()
    test_avg_pooling_forward_value()
    test_pooling_gradient_check()

