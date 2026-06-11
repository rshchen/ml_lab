import numpy as np
import matplotlib.pyplot as plt
from ml_lab.gradient import numerical_gradient_general

def target_function(x):
    """測試碗狀函數 f(x, y) = x^2 + y^2"""
    return np.sum(x**2)

def generate_trajectory(init_x, lr, step_num=20):
    """追蹤並紀錄特定學習率下的參數更新軌跡"""
    x = init_x.astype(np.float64, copy=True)
    traj = [x.copy()]
    
    for _ in range(step_num):
        grad = numerical_gradient_general(target_function, x)
        x = x - lr * grad
        # 邊界截斷防護：若發生嚴重發散，提早中斷迴圈避免繪圖數值溢出
        if np.any(np.abs(x) > 10): 
            break
        traj.append(x.copy())
        
    return np.array(traj)

def main():
    init_pos = np.array([-4.0, 4.0])
    steps = 20
    
    # 定義實驗對照組：過小、絕佳、過大
    lr_list = [0.01, 0.1, 0.95]
    titles = [
        "Learning Rate = 0.01\n(Under-shooting / Too Slow)",
        "Learning Rate = 0.1\n(Perfect Convergence)",
        "Learning Rate = 0.95\n(Over-shooting / Oscillation)"
    ]
    
    # 初始化 1 行 3 列的子圖畫布架構
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 建立網格地形數據
    x_val = np.linspace(-5, 5, 100)
    y_val = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_val, y_val)
    Z = X**2 + Y**2 
    
    for idx, lr in enumerate(lr_list):
        # 填空引導：動態指派當前迭代要操作的子圖軸物件 (Axes)
        ax = axes[idx]
        
        # 3.1 繪製背景等高線地形圖
        contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # 3.2 生成更新路徑
        traj = generate_trajectory(init_pos, lr, step_num=steps)
        
        # 利用折線圖 (plot) 在子圖上繪製紅色的離散更新軌跡路徑
        ax.plot(traj[:, 0], traj[:, 1], 'ro-', markersize=4, linewidth=1.5, label='SGD Path')
        
        # 標註起點與終點位置
        ax.plot(init_pos[0], init_pos[1], 'go', markersize=8, label='Start (-4, 4)')
        ax.plot(0, 0, 'bx', markersize=10, label='Global Min (0, 0)')
        
        # 3.3 圖表細節與美化
        ax.set_title(titles[idx], fontsize=12, fontweight='bold')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        ax.set_aspect('equal')
        

    plt.tight_layout()
    plt.suptitle("The Geometric Topology of Different Learning Rates (eta)", y=1.05, fontsize=16, fontweight='bold')
    
    # 輸出圖檔備查
    # plt.savefig("lr_comparison_topology.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

