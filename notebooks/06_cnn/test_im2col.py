import numpy as np
from ml_lab.cnn import im2col, col2im


def test_im2col_shape():
    """測試 im2col 輸出維度是否正確"""
    N, C, H, W = 2, 3, 4, 4
    FH, FW = 3, 3
    stride, padding = 1, 0

    x = np.random.randn(N, C, H, W)
    col = im2col(x, FH, FW, stride, padding)

    out_h = (H + 2 * padding - FH) // stride + 1
    out_w = (W + 2 * padding - FW) // stride + 1

    expected_shape = (N * out_h * out_w, C * FH * FW)
    assert col.shape == expected_shape, f"Expected {expected_shape}, got {col.shape}"
    print("test_im2col_shape 通過！")


def test_col2im_reconstruction():
    """測試 col2im 加算累積邏輯：使用全 1 矩陣驗證重疊區域累加次數"""
    N, C, H, W = 1, 1, 3, 3
    FH, FW = 2, 2
    stride, padding = 1, 0

    x_shape = (N, C, H, W)
    out_h = (H - FH) // stride + 1
    out_w = (W - FW) // stride + 1

    # 模擬全 1 的 dx_col (4 個視窗, 每個視窗 4 個像素)
    dummy_dx_col = np.ones((N * out_h * out_w, C * FH * FW))
    dx = col2im(dummy_dx_col, x_shape, FH, FW, stride, padding)

    # 在 3x3 輸入、2x2 Filter 下，中央 (1,1) 位置被 4 個視窗覆蓋，值應為 4.0
    expected_center_val = 4.0
    assert np.isclose(dx[0, 0, 1, 1], expected_center_val), f"Expected {expected_center_val}, got {dx[0, 0, 1, 1]}"
    print("test_col2im_reconstruction 重疊區域累加測試通過！")


if __name__ == "__main__":
    test_im2col_shape()
    test_col2im_reconstruction()

