import numpy as np
from ml_lab.dataset import load_mnist  

def test_mnist_pipeline():
    print("=== 開始發動數據管道幾何對帳 ===")
    
    # 呼叫你封裝好的管道
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot=True)
    
    # 防線一：檢驗標準幾何切片形狀 (Shape)
    print(f"x_train shape: {x_train.shape} | 預期: (60000, 784)")
    print(f"t_train shape: {t_train.shape} | 預期: (60000, 10)")
    print(f"x_test shape:  {x_test.shape}  | 預期: (10000, 784)")
    print(f"t_test shape:  {t_test.shape}  | 預期: (10000, 10)")
    
    assert x_train.shape == (60000, 784), "訓練集影像維度錯誤！"
    assert t_train.shape == (60000, 10),  "訓練集標籤維度錯誤！"
    assert x_test.shape == (10000, 784),  "測試集影像維度錯誤！"
    assert t_test.shape == (10000, 10),   "測試集標籤維度錯誤！"
    
    # 防線二：檢驗歸一化邊界值 (Value Range)
    print(f"x_train Max: {np.max(x_train)} | Min: {np.min(x_train)} | 預期: 1.0 附近與 0.0")
    assert np.max(x_train) <= 1.0 and np.min(x_train) >= 0.0, "歸一化地貌崩塌，數值超出 0~1 區間！"
    
    # 防線三：檢驗 One-Hot 陣列純淨度
    # 每一列的加總必須嚴格等於 1（代表幾率分佈完全歸一）
    print(f"t_train 行加總檢查: {np.all(np.sum(t_train, axis=1) == 1.0)} | 預期: True")
    assert np.all(np.sum(t_train, axis=1) == 1.0), "One-Hot 矩陣橫向加總不為 1，索引對齊失敗！"
    
    print("\n[PASS] 數據管道形狀、邊界值、One-Hot 索引全部對帳通過！100% 具備通車資格。")

if __name__ == "__main__":
    test_mnist_pipeline()

