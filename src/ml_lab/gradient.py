
import numpy as np


def numerical_diff(f, x, h=1e-4):
    """
    通用單變數數值微分 (中心差分)
    預設使用黃金步長 h = 1e-4，確保在雙精度浮點數下壓制截斷誤差至 O(h^2)
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient_general(f, x):
    """通用多維數值梯度計算器 (支援 1D, 2D 甚至 N 維張量)
    f: 黑盒子目標函數 (預期接收原始形狀的 x)
    x: 當前的參數點 NumPy 陣列 (可能為多維矩陣)
    """
    h = 1e-4
    
    # 紀錄原始幾何形狀以便最後還原
    original_shape = x.shape

    # 此處假設傳入的 x 在外部已是浮點數，確保 ravel 能安全產生連動 View
    # [ 重要 ] 透過 ravel 取得一維視圖（View），指向同一個記憶體區塊
    x_flat = x.ravel()
    grad_flat = np.zeros_like(x_flat)
    
    for idx in range(x_flat.size):
        tmp_val = x_flat[idx]
        
        # 透過視圖修改數值，實作中心差分的前向正向推動與 Loss 計算
        x_flat[idx] = tmp_val + h
        fxh1 = f(x)
        
        # 透過視圖修改數值，實作中心差分的逆向負向推動與 Loss 計算
        x_flat[idx] = tmp_val - h
        fxh2 = f(x)
        
        # 利用中心差分公式計算該軸向的偏導數逼近值
        grad_flat[idx] = (fxh1 - fxh2)/(2 * h)
        
        # 座標還原，透過視圖維持原始陣列其餘維度不變
        x_flat[idx] = tmp_val
        
    # 填空引導：將扁平化的梯度矩陣，還原成與輸入參數完全一致的幾何結構
    return grad_flat.reshape(original_shape)