import numpy as np
from ml_lab.layers import Dropout

def test_dropout():
    np.random.seed(42)
    print("=" * 60)
    print(" 啟動 Dropout 模組單元測試")
    print("=" * 60)

    # 1. 前向傳播與期望值測試
    x = np.ones((10000, 100)) # 大樣本評估統計量
    dropout_ratio = 0.4
    dropout_layer = Dropout(dropout_ratio=dropout_ratio)

    out_train = dropout_layer.forward(x, is_train=True)
    out_test = dropout_layer.forward(x, is_train=False)

    train_mean = np.mean(out_train)
    test_mean = np.mean(out_test)

    print(f"失活率 (Dropout Ratio) : {dropout_ratio}")
    print(f"訓練階段輸出期望值    : {train_mean:.4f} (目標 ≈ 1.0000)")
    print(f"推論階段輸出期望值    : {test_mean:.4f} (目標 = 1.0000)")

    assert np.isclose(train_mean, 1.0, atol=0.05), "Inverted Scaling 期望值未完美對齊！"
    print("能量守恆 (Expected Energy Conservation) 測試通過！")

    # 2. 反向傳播梯度測試
    dout = np.random.randn(5, 5)
    x_small = np.random.randn(5, 5)
    _ = dropout_layer.forward(x_small, is_train=True)
    dx = dropout_layer.backward(dout)

    # 檢查失活位置的梯度是否精確為 0
    zero_mask_match = (dx[~dropout_layer.mask] == 0).all()
    print(f"\n反向傳播失活神經元梯度切斷檢驗: {'通過' if zero_mask_match else '失敗'}")

    print("=" * 60)
    print(" Dropout 模組所有單元測試順利完成！")
    print("=" * 60)

if __name__ == "__main__":
    test_dropout()

