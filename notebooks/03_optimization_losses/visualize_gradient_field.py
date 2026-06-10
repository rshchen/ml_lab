import numpy as np
import matplotlib.pyplot as plt

def target_function(x0, x1):
    """測試碗狀函數: f(x0, x1) = x0^2 + x1^2"""
    return x0**2 + x1**2

def analytical_gradient(x0, x1):
    """梯度解析解: df/dx0 = 2x0, df/dx1 = 2x1"""
    return 2*x0, 2*x1

if __name__ == "__main__":
    # 1. 產生二維網格空間
    x0_range = np.linspace(-2.0, 2.0, 20)
    x1_range = np.linspace(-2.0, 2.0, 20)
    # 利用 NumPy 函式將一維座標軸拉伸組合為二維座標矩陣網路
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    # 2. 計算網格上每一點的真實 Loss 值與梯度向量
    Z = target_function(X0, X1)
    grad_X0, grad_X1 = analytical_gradient(X0, X1)
    
    # 為了具象化「梯度下降」，我們必須繪製梯度的反方向
    descend_X0 = -grad_X0
    descend_X1 = -grad_X1
    
    # 3. 開始繪製幾何地貌
    plt.figure(figsize=(8, 7))
    
    # 調用二維高度等高線繪圖函式，繪製底面座標與對應高度 Z 的關係
    contours = plt.contour(X0, X1, Z, levels=12, colors='gray', linewidths=0.8)
    plt.clabel(contours, inline=True, fontsize=8)
    
    # 調用二維向量箭頭場繪圖函式，傳入起點座標矩陣與對應的速度切向量分量
    plt.quiver(X0, X1, descend_X0, descend_X1, color='crimson', 
               angles='xy', scale_units='xy', scale=15, width=0.004,
               label='Negative Gradient (-\u2207f)')
    
    plt.plot(0, 0, 'go', markersize=10, label='Global Minimum (0, 0)')
    plt.title("Gradient Vector Field & Optimization Trajectory", fontsize=12)
    plt.xlabel("$x_0$ (Weight 1)", fontsize=10)
    plt.ylabel("$x_1$ (Weight 2)", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.axis('equal')
    
    print("正在生成梯度向量場幾何圖表...")
    plt.show()

