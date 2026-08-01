import numpy as np


def im2col(x: np.ndarray, FH: int, FW: int, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    將四維輸入張量 (N, C, H, W) 展開為二維矩陣 (N * OH * OW, C * FH * FW)
    """
    N, C, H, W = x.shape
    out_h = int((H + 2 * padding - FH) / stride) + 1
    out_w = int((W + 2 * padding - FW) / stride) + 1

    if padding > 0:
        x_padded = np.pad(
            x,
            pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant',
            constant_values=0
        )
    else:
        x_padded = x

    col = np.zeros((N, C, FH, FW, out_h, out_w))

    # TODO: 請填充 im2col 切片複製邏輯
    # 提示：寫兩重迴圈遍歷 Filter 的高 (FH) 與寬 (FW)
    # 透過步幅 stride 擷取整個 Batch 和所有 Channel 的空間區域
    for y in range(FH):
        y_max = y + stride * out_h
        for x_idx in range(FW):
            x_max = x_idx + stride * out_w
            # 填空：將切片賦值給 col 的對應位置
            # col[:, :, y, x_idx, :, :] = ...
            col[:, :, y, x_idx, :, :] = x_padded[:,:, y:y_max:stride, x_idx:x_max:stride]

    # TODO: 執行維度重排與展平
    # 提示 1: transpose 將軸向轉為 (N, out_h, out_w, C, FH, FW)
    # 提示 2: reshape 展平為 (N * out_h * out_w, -1)
    # col = ...
    return col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, C * FH * FW)


def col2im(col: np.ndarray, x_shape: tuple, FH: int, FW: int, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    將二維梯度矩陣 (N * OH * OW, C * FH * FW) 還原並累加回四維張量 (N, C, H, W)
    """
    N, C, H, W = x_shape
    out_h = int((H + 2 * padding - FH) / stride) + 1
    out_w = int((W + 2 * padding - FW) / stride) + 1

    # TODO: 將二維列矩陣 col 重構回六維張量 col_reshaped (N, C, FH, FW, out_h, out_w)
    # 提示：先 reshape(N, out_h, out_w, C, FH, FW) 再 transpose(0, 3, 4, 5, 1, 2)
    col_reshaped = col.reshape(N, out_h, out_w, C, FH, FW).transpose(0, 3, 4, 5, 1, 2)

    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=col.dtype)

    # TODO: 使用對稱切片執行重疊區域梯度的累加 (+=)
    for y in range(FH):
        y_max = y + stride * out_h
        for x_idx in range(FW):
            x_max = x_idx + stride * out_w
            # 填空：利用切片將 col_reshaped 中的梯度累加回 x_padded
            # x_padded[:, :, y:y_max:stride, x_idx:x_max:stride] += ...
            x_padded[:, :, y:y_max:stride, x_idx:x_max:stride] += col_reshaped[:, :, y, x_idx, :, :]

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded

