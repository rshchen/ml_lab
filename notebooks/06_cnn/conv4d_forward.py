import numpy as np


def conv4d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray = None,
    padding: int = 0,
    stride: int = 1
) -> np.ndarray:
    """
    多通道四維張量卷積前向傳播實作
    
    x: 輸入數據，形狀為 (N, C, H, W)
    weight: 權重 (Filters)，形狀為 (FN, C, FH, FW)
    bias: 偏置項，形狀為 (FN,)
    padding: 零填充大小 (P)
    stride: 移動步幅 (S)
    """
    N, C, H, W = x.shape
    FN, C_filter, FH, FW = weight.shape
    
    # 斷言檢查：輸入通道數必須與 Filter 深度一致
    assert C == C_filter, f"Input channels ({C}) must match Filter channels ({C_filter})"
    
    # -------------------------------------------------------------------------
    # 計算輸出特徵圖的尺寸 OH 與 OW
    # -------------------------------------------------------------------------
    # TODO: 計算 out_h 與 out_w (請轉為 int)
    out_h = int(((H + 2 * padding - FH) / stride) + 1)
    out_w = int(((W + 2 * padding - FW) / stride) + 1)
    
    # -------------------------------------------------------------------------
    # 執行 Zero-Padding
    # 提示: 僅在空間軸 (H, W) 進行填充，Batch (N) 與 Channel (C) 軸補零量為 0
    # pad_width 格式: ((pad_N_left, pad_N_right), (pad_C_left, pad_C_right), ...)
    # -------------------------------------------------------------------------
    if padding > 0:
        # TODO: 使用 np.pad 對 x 的空間軸 H, W 進行 2D zero-padding
        x_padded = np.pad(
            x, 
            pad_width=((0,0), (0,0), (padding, padding), (padding, padding)), 
            mode='constant', 
            constant_values=0
        )
    else:
        x_padded = x
        
    # 初始化四維輸出張量 (N, FN, OH, OW)
    out = np.zeros((N, FN, out_h, out_w))
    
    # -------------------------------------------------------------------------
    # 四重迴圈進行多通道張量加權求和與偏置廣播
    # -------------------------------------------------------------------------
    for n in range(N):                   # 遍歷 Batch 樣本
        for fn in range(FN):             # 遍歷各個 Filter (Output Channel)
            for y in range(out_h):       # 遍歷垂直空間軸
                for x_idx in range(out_w): # 遍歷水平空間軸
                    
                    # TODO: 計算當前視窗在空間軸上的切片座標
                    y_start = y * stride
                    y_end = y_start + FH
                    x_start = x_idx * stride
                    x_end = x_start + FW
                    
                    # TODO: 擷取 3D 局部區域 window，形狀應為 (C, FH, FW)
                    # 提示: 從 x_padded 中切出第 n 個樣本、所有通道、以及對應空間視窗
                    window = x_padded[n, :, y_start:y_end, x_start:x_end]

                    
                    # TODO: 進行元素對應相乘後求和 (所有通道融合為單點純量)
                    conv_sum = np.sum(window * weight[fn])
                    
                    # TODO: 若存在 bias，加入對應 Filter (fn) 的偏置項
                    if bias is not None:
                        conv_sum += bias[fn]  # 請修正此處
                        
                    out[n, fn, y, x_idx] = conv_sum
                    
    return out


if __name__ == "__main__":
    # 測試資料驗證 (N=2, C=3, H=4, W=4)
    np.random.seed(42)
    x_dummy = np.random.randn(2, 3, 4, 4)

    # 建立 2 個 Filter (FN=2, C=3, FH=3, FW=3)
    weight_dummy = np.random.randn(2, 3, 3, 3)
    bias_dummy = np.array([0.5, -0.5])

    output_map = conv4d_forward(
        x_dummy, 
        weight_dummy, 
        bias=bias_dummy, 
        padding=1, 
        stride=1
    )

    print("輸入張量形狀 (N, C, H, W)       :", x_dummy.shape)
    print("權重張量形狀 (FN, C, FH, FW)   :", weight_dummy.shape)
    print("偏置張量形狀 (FN,)              :", bias_dummy.shape)
    print("輸出張量形狀 (N, FN, OH, OW)   :", output_map.shape)

