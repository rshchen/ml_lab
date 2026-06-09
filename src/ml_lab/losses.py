import numpy as np
from sklearn.datasets import fetch_openml
from ml_lab.activations import sigmoid, softmax

def mean_squared_error(y, t):
    """
    均方誤差 (Mean Squared Error) - 專治【回歸問題】
    支援單筆 1D 向量與 Batch 2D 矩陣
    
    物理維度與格式限制：
    1. 本函數「不支援」未轉換的分類數字標籤 (如 [2, 0, 4])。
    2. 真實標籤 t 必須是與預測資料 y 格式完全一致的「浮點數矩陣」或「One-Hot 矩陣」。
    3. 在回歸任務中，t 的數值代表真實的實數連續目標（如房價、坐標）。
    """
    # 升維防線：強制保持內部 sample_losses 永遠為一維陣列的形狀守恆思維
    if y.ndim == 1:
        y = y.reshape(1, -1)
        t = t.reshape(1, -1)
        
    # 沿著橫向（欄位軸）加總每個樣本各自的平方誤差 -> sample_losses 永遠是 (Batch,)
    sample_losses = 0.5 * np.sum((y - t) ** 2, axis=1)
    
    # 對所有樣本的 Loss 取平均值並回傳純量
    return np.mean(sample_losses)


def cross_entropy_error(y, t):
    """
    交叉熵誤差 (Cross Entropy Error) - 專治【分類問題】
    同時支援 One-Hot 編碼與傳統數字標籤
    """
    delta = 1e-7
    
    # 單筆資料分流防線：保持最純粹的向量與純量操作
    if y.ndim == 1:
        # 情況一：t 是 One-Hot 編碼 (形狀與 y 完全一致)
        if t.size == y.size:
            return -np.sum(t * np.log(y + delta))
        # 情況二：t 是傳統數字標籤 (形狀是 1D 陣列)
        else:
            return -np.log(y[t] + delta)
            
    # 批次資料分流：啟動高階 Batch 處理機制
    batch_size = y.shape[0]
    
    # 情況一：t 是 One-Hot 編碼 (形狀與 y 完全一致)
    if t.size == y.size:
        # 相乘後全加總，並除以 batch_size
        return -np.sum(t * np.log(y + delta)) / batch_size
        
    # 情況二：t 是傳統數字標籤 (形狀是 1D 陣列)
    else:
        # 利用 np.arange(batch_size) 搭配 t 作為欄位索引
        return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

