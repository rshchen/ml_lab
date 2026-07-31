import numpy as np


def pool4d_forward(
    x: np.ndarray,
    pool_size: tuple = (2, 2),
    stride: int = 2,
    mode: str = 'max'
) -> np.ndarray:
    """
    多通道四維張量池化層前向傳播實作
    
    x: 輸入數據，形狀為 (N, C, H, W)
    pool_size: 池化視窗尺寸 (PH, PW)
    stride: 移動步幅 (S)
    mode: 池化模式，可選 'max' 或 'avg'
    """
    N, C, H, W = x.shape
    PH, PW = pool_size
    
    # -------------------------------------------------------------------------
    # [Task 1] 計算輸出特徵圖的尺寸 OH 與 OW
    # -------------------------------------------------------------------------
    # TODO: 計算 out_h 與 out_w (請轉為 int)
    out_h = int(((H - PH) / stride) + 1)
    out_w = int(((W - PW) / stride) + 1)
    
    # 初始化四維輸出張量 (N, C, OH, OW) - 注意通道數仍為 C
    out = np.zeros((N, C, out_h, out_w))
    
    # -------------------------------------------------------------------------
    # [Task 2] 四重迴圈進行張量空間切片與統計摘要
    # -------------------------------------------------------------------------
    for n in range(N):                   # 遍歷 Batch 樣本
        for c in range(C):               # 遍歷各個 Channel (獨立池化，不跨通道融合)
            for y in range(out_h):       # 遍歷垂直空間軸
                for x_idx in range(out_w): # 遍歷水平空間軸
                    
                    # TODO: 計算當前視窗在空間軸上的切片座標
                    y_start = y * stride
                    y_end = y_start + PH
                    x_start = x_idx * stride
                    x_end = x_start + PW
                    
                    # TODO: 擷取第 n 個樣本、第 c 個通道的 2D 局部區域 window (形狀為 PH x PW)
                    window = x[n, c, y_start:y_end, x_start:x_end]
                    
                    # TODO: 依據 mode 執行對應的統計運算 (np.max 或 np.mean)
                    if mode == 'max':
                        out[n, c, y, x_idx] = np.max(window)
                    elif mode == 'avg':
                        out[n, c, y, x_idx] = np.mean(window)
                    else:
                        raise ValueError(f"Unsupported pooling mode: {mode}")
                        
    return out


if __name__ == "__main__":
    # 測試資料驗證 (N=2, C=3, H=4, W=4)
    np.random.seed(42)
    x_dummy = np.random.randn(2, 3, 4, 4)

    max_pooled = pool4d_forward(x_dummy, pool_size=(2, 2), stride=2, mode='max')
    avg_pooled = pool4d_forward(x_dummy, pool_size=(2, 2), stride=2, mode='avg')

    print("輸入張量形狀 (N, C, H, W) :", x_dummy.shape)
    print("最大池化輸出形狀 (N, C, OH, OW) :", max_pooled.shape)
    print("平均池化輸出形狀 (N, C, OH, OW) :", avg_pooled.shape)

