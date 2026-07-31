import numpy as np


def conv2d_forward(x: np.ndarray, kernel: np.ndarray, padding: int = 0, stride: int = 1) -> np.ndarray:
    """
    通用單通道二維卷積前向傳播實作 (支援 Padding 與 Stride)
    
    x: 輸入特徵圖，形狀為 (H, W)
    kernel: 卷積核，形狀為 (FH, FW)
    padding: 零填充大小 (P)
    stride: 移動步幅 (S)
    """
    H, W = x.shape
    FH, FW = kernel.shape
    
    # -------------------------------------------------------------------------
    # 執行 Zero-Padding 擴充
    # 提示: 使用 np.pad 在空間軸 (H, W) 兩側補零
    # -------------------------------------------------------------------------
    if padding > 0:
        # 使用 np.pad 對 x 進行 2D zero-padding
        x_padded = np.pad(
            x, 
            pad_width=((padding, padding), (padding, padding)), 
            mode='constant', 
            constant_values=0
        )
    else:
        x_padded = x
        
    # -------------------------------------------------------------------------
    # 計算輸出特徵圖的尺寸 OH 與 OW
    # 公式: OH = ((H + 2P - FH) / S) + 1
    # -------------------------------------------------------------------------
    # 計算 out_h 與 out_w (請轉為 int)
    out_h = int(((H + 2 * padding - FH) / stride) + 1)
    out_w = int(((W + 2 * padding - FW) / stride) + 1)
    
    # 初始化輸出特徵圖
    output = np.zeros((out_h, out_w))
    
    # -------------------------------------------------------------------------
    # 依照 Stride 步幅進行滑動視窗加權求和
    # -------------------------------------------------------------------------
    for y in range(out_h):
        for x_idx in range(out_w):
            # 計算當前視窗在 x_padded 上的切片座標 (y_start, y_end, x_start, x_end)
            y_start = y * stride
            y_end = y_start + FH
            x_start = x_idx * stride
            x_end = x_start + FW
            
            # 擷取局部區域 (window) 並進行元素對應相乘後求和 (Element-wise Summation)
            window = x_padded[y_start:y_end, x_start:x_end]
            output[y, x_idx] = np.sum(window * kernel)
            
    return output


if __name__ == "__main__":
    # 測試資料驗證
    input_map = np.array([
        [1, 2, 0, 1],
        [0, 1, 3, 2],
        [2, 0, 1, 0],
        [1, 1, 0, 2]
    ], dtype=float)

    test_kernel = np.array([
        [1, 0],
        [0, -1]
    ], dtype=float)

    P = 1
    S = 2

    feature_map = conv2d_forward(input_map, test_kernel, padding=P, stride=S)

    print("原始輸入尺寸:", input_map.shape)
    print("卷積核尺寸:", test_kernel.shape)
    print(f"配置參數: Padding={P}, Stride={S}")
    print("計算後輸出尺寸:", feature_map.shape)
    print("\n輸出 Feature Map 內容:\n", feature_map)

