import numpy as np
from ml_lab.activations import sigmoid, relu, softmax
from ml_lab.losses import cross_entropy_error
from ml_lab.cnn import im2col, col2im

class AffineLayer:
    def __init__(self):
        self.x = None
        self.w = None
        self.b = None
    def forward(self, x, w, b):
        self.x = x
        self.w = w
        self.b = b
        out = x @ w + b
        return out 
    def backward(self, dout):
        dx = dout @ self.w.T
        dw = self.x.T @ dout
        db = dout.sum(axis=0)
        return dx, dw, db
    
class ReluLayer:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x<=0)
        return relu(x)
    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx
    
class SigmoidLayer:
    def __init__(self):
        self.y = None
    def forward(self, x):
        out = self.y = sigmoid(x)
        return out
    def backward(self, dout):
        y = self.y
        dx = dout * y * (1 - y)
        return dx
    
class MeanSquaredErrorLayer:
    """
    均方誤差 (Mean Squared Error) - 專治【回歸問題】
    支援單筆 1D 向量與 Batch 2D 矩陣
    
    物理維度與格式限制：
    1. 本函數「不支援」未轉換的分類數字標籤 (如 [2, 0, 4])。
    2. 真實標籤 t 必須是與預測資料 y 格式完全一致的「浮點數矩陣」或「One-Hot 矩陣」。
    3. 在回歸任務中，t 的數值代表真實的實數連續目標（如房價、坐標）。

    為了不重複升維防線，因此這邊沒有直接 from ml_lab.losses import mean_squared_error
    """
    def __init__(self):
        self.y = None
        self.t = None
    def forward(self, y, t):
        # 升維防線
        if y.ndim == 1:
            y = y.reshape(1, -1)
            t = t.reshape(1, -1)
        self.y = y
        self.t = t
        sample_losses = 0.5 * np.sum((y - t) ** 2, axis=1)

        out = np.mean(sample_losses)
        return out
    def backward(self, dout):
        # 此時的 self.y 必定是 (Batch, Feature) 的 2D 矩陣
        batch_size = self.y.shape[0]
        dy = (self.y - self.t)/batch_size
        return dy * dout

class SoftmaxWithCeeLayer:
    def __init__(self):
        self.y = None
        self.t = None
    def forward(self, x, t):
        self.t = t
        self.y = y = softmax(x)
        loss = cross_entropy_error(y, t)
        return loss
    def backward(self, dout=1.0):
        y = self.y
        t = self.t
        batch_size = 1 if y.ndim == 1 else y.shape[0]
        
        # 只要不是 One-Hot (尺寸不相等)，就代表是傳統數字標籤
        if t.size != y.size:
            t_one_hot = np.zeros_like(y)
            # 當 y 是 1D (單筆) 時，batch_size=1，t_one_hot[t] 直接利用 1D 索引變為 1
            # 當 y 是 2D (批次) 時，利用 np.arange 進行二維進階索引
            if y.ndim == 1:
                t_one_hot[t] = 1
            else:
                t_one_hot[np.arange(batch_size), t] = 1
            t = t_one_hot
            
        dx = (y - t) / batch_size
        return dx * dout


class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, eps=1e-5):
        self.gamma = gamma          # 縮放參數 (D,)
        self.beta = beta            # 平移參數 (D,)
        self.momentum = momentum    # 移動平均衰減率 (EMA)
        self.eps = eps              # 數值穩定項

        # 推論時使用的全域統計量 (Running Mean / Var)
        self.running_mean = None
        self.running_var = None

        # 反向傳播快取
        self.cache = None

    def forward(self, x, is_train=True):
        N, D = x.shape
        
        if self.running_mean is None:
            self.running_mean = np.zeros(D)
            self.running_var = np.ones(D)

        if is_train:
            # 計算當前 Batch 的均值與中心化矩陣 xc
            mu = np.mean(x, axis=0)
            xc = x - mu

            # 計算當前 Batch 的方差與標準差
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + self.eps)

            # 計算標準化特徵 xn 與縮放平移的最終輸出 out
            xn = xc / std
            out = self.gamma * xn + self.beta

            # 更新全域指數移動平均 (EMA) 統計量
            # 若 self.momentum = 0.9 (Decay rate)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.cache = (xc, xn, std, N)
        else:
            # 填寫推論階段的中心化與標準化流程
            xc = x - self.running_mean
            xn = xc / np.sqrt(self.running_var + self.eps)
            out = self.gamma * xn + self.beta

        return out

    def backward(self, dout):
        xc, xn, std, N = self.cache

        # 計算對可學習參數 gamma 與 beta 的梯度 dgamma, dbeta
        dgamma = np.sum(dout * xn, axis=0)
        dbeta = np.sum(dout, axis=0)

        # 套用簡化鏈鎖律公式計算對輸入 x 的梯度 dx
        # dx = (gamma / (N * std)) * (N * dout - sum(dout) - xn * sum(dout * xn))
        dx = (self.gamma / (N * std)) * (N * dout - np.sum(dout, axis=0) - xn * np.sum(dout * xn, axis=0))

        return dx, dgamma, dbeta


class Dropout:
    """
    Inverted Dropout 實作
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, is_train=True):
        if is_train:
            # 生成伯努利隨機遮罩 mask
            # 提示: 使用 np.random.rand(*x.shape) 生成 [0, 1) 的均勻分佈，並判斷是否大於 dropout_ratio
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            
            # 執行 Inverted Scaling 反向縮放前向傳播
            # 提示: 將輸入 x 與 mask 逐元素相乘，並除以保留機率 (1 - self.dropout_ratio)
            return (x * self.mask) / (1 - self.dropout_ratio)
        else:
            # 推論階段直接 Pass 輸出
            return x

    def backward(self, dout):
        # 鏈鎖律反向傳播
        # 沿用前向傳播保存的 self.mask，並除以 (1 - self.dropout_ratio)
        return (dout * self.mask) / (1 - self.dropout_ratio)



class Convolution:
    """基於 im2col 與 col2im 的高效四維卷積層類別封裝"""

    def __init__(self, weight: np.ndarray, bias: np.ndarray = None, stride: int = 1, padding: int = 0):
        self.W = weight  # 形狀: (FN, C, FH, FW)
        self.b = bias    # 形狀: (FN,)
        self.stride = stride
        self.padding = padding

        # 前向傳播中間狀態快取 (用於反向傳播)
        self.x = None
        self.x_col = None
        self.W_row = None

        # 參數梯度快取
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        N, C, H, W = x.shape
        FN, C_filter, FH, FW = self.W.shape

        assert C == C_filter, f"Input channels ({C}) must match Filter channels ({C_filter})"

        out_h = int((H + 2 * self.padding - FH) / self.stride) + 1
        out_w = int((W + 2 * self.padding - FW) / self.stride) + 1

        # TODO 1: 呼叫 im2col 將輸入張量矩陣化 -> self.x_col
        self.x_col = im2col(x, FH, FW, self.stride, self.padding)

        # TODO 2: 將權重矩陣化為二維列矩陣 -> self.W_row (FN, C * FH * FW)
        self.W_row = self.W.reshape(FN, C * FH * FW)

        # TODO 3: GEMM 矩陣乘法 (XW 模式) 並加總 bias
        # 提示：Out_mat = X_col @ W_row.T + b
        out_mat = self.x_col @ self.W_row.T
        if self.b is not None:
            out_mat += self.b

        # TODO 4: 還原四維張量形狀 (N, FN, out_h, out_w)
        # 提示：先 reshape(N, out_h, out_w, FN) 再 transpose(0, 3, 1, 2)
        out = out_mat.reshape(N, out_h, out_w, FN).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        FN, C, FH, FW = self.W.shape

        # TODO 1: 轉置並展開上游傳回的梯度 dout -> dout_mat (N * out_h * out_w, FN)
        # 提示：dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        dout_mat = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # TODO 2: 計算偏置梯度 db (對 Batch 與空間維度求和)
        if self.b is not None:
            self.db = np.sum(dout_mat, axis=0)

        # TODO 3: 套用全連接層規則求權重矩陣梯度 dW_mat，並 reshape 為 (FN, C, FH, FW)
        # 提示：dW_mat = dout_mat.T @ self.x_col
        dW_mat = dout_mat.T @ self.x_col
        self.dW = dW_mat.reshape(FN, C, FH, FW)

        # TODO 4: 套用全連接層規則求輸入展開矩陣梯度 dx_col
        # 提示：dx_col = dout_mat @ self.W_row
        dx_col = dout_mat @ self.W_row

        # TODO 5: 呼叫 col2im 還原並累加梯度回 dx
        dx = col2im(dx_col, self.x.shape, FH, FW, self.stride, self.padding)
        return dx


class Pooling:
    """基於 im2col 與 col2im 的四維池化層（支援 Max 與 Avg 模式）"""

    def __init__(self, pool_h: int, pool_w: int, stride: int = 2, padding: int = 0, mode: str = 'max'):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.padding = padding
        self.mode = mode.lower()

        assert self.mode in ['max', 'avg'], "Mode 必須為 'max' 或 'avg'"

        self.x = None
        self.arg_max = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        N, C, H, W = x.shape

        out_h = int((H + 2 * self.padding - self.pool_h) / self.stride) + 1
        out_w = int((W + 2 * self.padding - self.pool_w) / self.stride) + 1

        # TODO 1: 呼叫 im2col 將輸入張量展開 -> col (N * out_h * out_w, C * pool_h * pool_w)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.padding)

        # TODO 2: 將通道獨立隔離，重構維度 -> col_reshaped (N * out_h * out_w * C, pool_h * pool_w)
        # 提示：reshape 為 (-1, pool_h * pool_w)
        col_reshaped = col.reshape(-1, self.pool_h * self.pool_w)

        # TODO 3: 沿著視窗軸 (axis=1) 進行降維計算
        # 若為 'max' 模式：計算最大值 out_flat 並記錄索引 self.arg_max (使用 np.argmax)
        # 若為 'avg' 模式：計算平均值 out_flat (使用 np.mean)
        if self.mode == 'max':
            out_flat = np.max(col_reshaped, axis=1)
            self.arg_max = np.argmax(col_reshaped, axis=1)
        elif self.mode == 'avg':
            out_flat = np.mean(col_reshaped, axis=1)

        # TODO 4: 將降維後的結果還原為四維張量 (N, C, out_h, out_w)
        # 提示：先 reshape(N, out_h, out_w, C) 再 transpose(0, 3, 1, 2)
        out = out_flat.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, C, H, W = self.x.shape
        out_h = dout.shape[2]
        out_w = dout.shape[3]

        # TODO 1: 將上游傳回的梯度 dout 轉置並展平為一維陣列 -> dout_flat (N * out_h * out_w * C,)
        # 提示：dout.transpose(0, 2, 3, 1).flatten()
        dout_flat = dout.transpose(0, 2, 3, 1).flatten()

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout_flat.size, pool_size), dtype=dout.dtype)
        # TODO 2: 根據模式路由與分配梯度到 dmax 矩陣中
        # 若為 'max' 模式：利用 self.arg_max 將 dout_flat 填入對應的最大值位置
        # 若為 'avg' 模式：將 dout_flat 平均分配給池化視窗內的每一個點
        if self.mode == 'max':
            dmax[np.arange(self.arg_max.size), self.arg_max] = dout_flat
        elif self.mode == 'avg':
            dmax[:] = dout_flat[:, np.newaxis] / pool_size #這邊要切片複製才可以產生廣播

        # TODO 3: 將 dmax 重構為 im2col 的輸出形狀 dcol (N * out_h * out_w, C * pool_size)
        dcol = dmax.reshape(N * out_h * out_w, C * pool_size)

        # TODO 4: 呼叫 col2im 還原梯度回 dx
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.padding)
        
        return dx

