import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def target_function(x0, x1):
    return x0**2 + x1**2

if __name__ == "__main__":
    # 1. 建立二維網格空間 (完全對齊 2D 實驗的代數結構)
    x0_range = np.linspace(-2.0, 2.0, 30)
    x1_range = np.linspace(-2.0, 2.0, 30)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    Z = target_function(X0, X1)
    
    # 2. 開啟 3D 投影畫布
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'})
    
    # 3. 繪製 3D 表面圖 (Surface Plot)
    # cm.coolwarm 色調會將碗底低處染成藍色 (低 Loss)，高處染成紅色 (高 Loss)
    surf = ax.plot_surface(X0, X1, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True, alpha=0.8)
    
    # 4. 幾何視覺對齊：在 3D 空間底部投影出 2D 等高線
    # zdir='z' 代表將等高線沿著 Z 軸往下拍扁，offset=0 代表投影在 Z=0 的平面上
    ax.contour(X0, X1, Z, zdir='z', offset=0, cmap=cm.coolwarm, linewidths=1.5)
    
    # 5. 設置 3D 空間座標軸與視角
    ax.set_title("3D Loss Surface & 2D Contour Projection", fontsize=12)
    ax.set_xlabel("$x_0$ (Weight 1)")
    ax.set_ylabel("$x_1$ (Weight 2)")
    ax.set_zlabel("Loss ($z$)")
    ax.set_zlim(0, 8)
    
    # 調整視角 (Elevation 仰角, Azimuth 方位角)，以最利於觀察斜率的角度切入
    ax.view_init(elev=25, azim=-45)
    
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Loss Value')
    print("正在生成 3D Loss 碗狀曲面幾何圖表...")
    plt.show()

