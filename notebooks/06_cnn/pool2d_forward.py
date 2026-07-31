import numpy as np


def pool2d_forward(
    x: np.ndarray, 
    pool_size: tuple = (2, 2), 
    stride: int = 2, 
    mode: str = 'max'
) -> np.ndarray:
    """
    單通道二維池化層前向傳播實作
    
    x: 輸入特徵圖，形狀為 (H, W)
    pool_size: 池化視窗尺寸 (PH, PW)
    stride: 移動步幅 (S)
    mode: 池化模式，可選 'max' 或 'avg'
    """
    H, W = x.shape
    PH, PW = pool_size
    
    # -------------------------------------------------------------------------
    # [Task 1] 計算輸出特徵圖的尺寸 OH 與 OW
    # 公式: OH = ((H - PH) / S) + 1
    # -------------------------------------------------------------------------
    # TODO: 計算 out_h 與 out_w (請轉為 int)
    out_h = int(((H - PH) / stride) + 1)
    out_w = int(((W - PW) / stride) + 1)
    
    # 初始化輸出特徵圖
    output = np.zeros((out_h, out_w))
    
    # -------------------------------------------------------------------------
    # [Task 2] 依照 Stride 步幅進行滑動視窗統計摘要
    # -------------------------------------------------------------------------
    for y in range(out_h):
        for x_idx in range(out_w):
            # TODO: 計算當前視窗在輸入矩陣上的切片座標
            y_start = y * stride
            y_end = y_start + PH
            x_start = x_idx * stride
            x_end = x_start + PW
            
            # TODO: 切出 2D 局部區域視窗 (window)
            window = x[y_start:y_end, x_start:x_end]
            
            # TODO: 依據 mode 執行對應的統計運算 (np.max 或 np.mean)
            if mode == 'max':
                output[y, x_idx] = np.max(window)
            elif mode == 'avg':
                output[y, x_idx] = np.mean(window)
            else:
                raise ValueError(f"Unsupported pooling mode: {mode}")
                
    return output


if __name__ == "__main__":
    # 測試資料驗證
    input_map = np.array([
        [1, 3, 2, 1],
        [2, 8, 0, 6],
        [0, 4, 3, 2],
        [1, 7, 9, 5]
    ], dtype=float)

    max_pooled = pool2d_forward(input_map, pool_size=(2, 2), stride=2, mode='max')
    avg_pooled = pool2d_forward(input_map, pool_size=(2, 2), stride=2, mode='avg')

    print("原始輸入特徵圖 (4x4):\n", input_map)
    print("\n最大池化結果 (2x2):\n", max_pooled)
    print("\n平均池化結果 (2x2):\n", avg_pooled)

