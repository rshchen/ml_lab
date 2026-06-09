import matplotlib.pyplot as plt
import numpy as np

def mean_squared_error(y, t):
    """
    均方誤差 (Mean Squared Error)
    y: 神經網路的預測輸出 (NumPy 陣列)
    t: 真實的正確標籤 (NumPy 陣列)
    """
    # 實作 1/2 * sum((y - t)^2)
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    """
    交叉熵誤差 (Cross Entropy Error)
    y: Softmax 吐出的預測機率分佈 (NumPy 陣列)
    t: One-Hot 編碼的真實標籤 (NumPy 陣列)
    """
    # 防禦性設計：定義微小值，防止 np.log(0) 觸發 -inf 導致計算潰縮
    delta = 1e-7
    
    # 實作 -sum(t * log(y + delta))
    return -np.sum(t * np.log(y + delta))

if __name__ == "__main__":
    print("========================================")
    print(" losses.py 內建模組功能測試")
    print("========================================")
    
    # 1. 測試均方誤差 (MSE)
    # 假設正確答案是 2，模型猜 2 的成分很高
    t_dummy = np.array([0, 0, 1, 0, 0])
    y_dummy1 = np.array([0.1, 0.05, 0.7, 0.1, 0.05])
    y_dummy2 = np.array([0.1, 0.4, 0.1, 0.3, 0.1])
    
    print(f"模型猜得好時的 MSE : {mean_squared_error(y_dummy1, t_dummy):.5f}")
    print(f"模型猜得很爛時的 MSE: {mean_squared_error(y_dummy2, t_dummy):.5f}")
    
    # 測試交叉熵誤差 (Cross Entropy)
    print(f"模型猜得好時的 Cross Entropy : {cross_entropy_error(y_dummy1, t_dummy):.5f}")
    print(f"模型猜得很爛時的 Cross Entropy: {cross_entropy_error(y_dummy2, t_dummy):.5f}")
    
    # 幾何視覺化：親手畫出交叉熵的懲罰懸崖
    print("\n正在生成交叉熵損失函數的幾何圖像...")
    
    # 建立 X 軸，範圍自 0.001 到 1.0，切出 500 個點
    x = np.linspace(0.001, 1.0, 500)
    
    # 計算 Y 軸，計算交叉熵的核心公式 -log(x)
    y_cross_entropy = -np.log(x)
    
    plt.figure(figsize=(9, 5))
    
    # 使用 plt.plot 畫出這條紅色的懲罰曲線
    plt.plot(x, y_cross_entropy, label="y = -log(x)", color="tab:red", linewidth=2.5)
    
    # 標註完美預測點 (x=1.0, y=0.0)
    plt.scatter(1.0, 0.0, color="green", s=100, zorder=5)
    
    # 裝飾圖表
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("The Geometric Landscape of Cross Entropy Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Probability (y_i) -> Model's Confidence")
    plt.ylabel("Loss Value (E) -> Penalty Amount")
    plt.legend()
    
    print("圖表視窗即將開啟，請親眼見證紅色曲線在 X 趨近於 0 時衝向無窮大的恐怖懸崖！")
    plt.show()

