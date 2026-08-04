import numpy as np
from ml_lab.layers import AffineLayer, ReluLayer, SoftmaxWithCeeLayer, Pooling, Convolution

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init='he'):
        """
        TwoLayerNet 類別實作 (支援動態方差守恆初始化策略)
        
        Parameters:
        -----------
        input_size : int
            輸入層維度 (n_in_bottom)
        hidden_size : int
            隱藏層維度
        output_size : int
            輸出層維度 (n_out_top)
        weight_init : str or float
            初始化策略 ('he', 'xavier', 或自訂常數 std，如 0.01)
        """
        # 1. 權重參數主權一元化：通通鎖死在 self.params 字典中
        self.params = {}
    
        # 依據傳入策略計算各層權重之標準差 std (std = sqrt(Var))
        if str(weight_init).lower() == 'he':
            # He 初始化 (Kaiming Init): Var(W) = 2 / n_in
            std_w1 = np.sqrt(2.0 / input_size)
            std_w2 = np.sqrt(2.0 / hidden_size)
        elif str(weight_init).lower() == 'xavier':
            # Xavier 初始化: Var(W) = 1 / n_in
            std_w1 = np.sqrt(1.0 / input_size)
            std_w2 = np.sqrt(1.0 / hidden_size)
        else:
            # 純量常數初始化 (如原本的 0.01)
            std_w1 = float(weight_init)
            std_w2 = float(weight_init)
    


        # 套用具備方差守恆補償的權重初始化
        self.params['W1'] = std_w1 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std_w2 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
        # 2. 【組裝純淨無狀態的算子元件】（如同 PyTorch nn.Module 宣告屬性）
        # ===== 在這裡建立獨立的算子實例 =====
        self.affine1    = AffineLayer()
        self.relu1      = ReluLayer()
        self.affine2    = AffineLayer()
        
        # =========================================================================
        
        # 3. 終點節點：將 Softmax 與 CCE 的大合體代數融化節點單獨擺放
        self.last_layer = SoftmaxWithCeeLayer()
        
    def predict(self, x):
        # ===== 依序「指名道姓」向前傳導，並將參數動態注入變換器 =====
        x = self.affine1.forward(x, self.params['W1'], self.params['b1'])
        x = self.relu1.forward(x)
        x = self.affine2.forward(x, self.params['W2'], self.params['b2'])
        # =================================================================================
        return x
        
    def loss(self, x, t):
        score = self.predict(x)
        return self.last_layer.forward(score, t)
        
    def accuracy(self, x, t):
        score = self.predict(x)
        y = np.argmax(score, axis=1)
        # one hot
        if t.ndim != 1: 
            t = np.argmax(t, axis=1)

        batch_size = x.shape[0]
        return np.sum(y == t) / batch_size
        
    def gradient(self, x, t):
        # 1. 發動前向傳播，強迫各層內部節點動態記憶激活狀態與權重的矩陣
        self.loss(x, t)
        
        # 2. 收集梯度的容器
        grads = {}
        
        # 3. 初始梯度發動：大合體終點節點輸出 (Y - T) / M
        dout = 1.0
        dout = self.last_layer.backward(dout) 
        
        # 4. 【反向串接與精準攔截】
        # ===== 依序由後往前呼叫 backward，並當場攔截吐出的參數梯度 =====
        dout, grads['W2'], grads['b2'] = self.affine2.backward(dout)
        dout = self.relu1.backward(dout)
        _, grads['W1'], grads['b1'] = self.affine1.backward(dout)

        return grads


class SimpleConvNet:
    """
    經典 SimpleConvNet 類別實作 (無狀態算子風格，動態維度重構)
    架構：Conv -> ReLU -> Pool -> Affine -> ReLU -> Affine -> SoftmaxWithLoss
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num': 30, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 pool_param={'pool_h': 2, 'pool_w': 2, 'pad': 0, 'stride': 2},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        
        # 1. 解構 Conv 參數
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        # 2. 解構 Pool 參數
        pool_h = pool_param['pool_h']
        pool_w = pool_param['pool_w']
        pool_pad = pool_param['pad']
        pool_stride = pool_param['stride']
        
        input_size = input_dim[1]
        
        # 套用標準公式動態計算尺寸 (徹底消除除以 2 的魔術數字)
        conv_output_size = (input_size - filter_size + 2 * filter_pad) // filter_stride + 1
        self.pool_output_size = (conv_output_size - pool_h + 2 * pool_pad) // pool_stride + 1
        pool_output_pixel = filter_num * self.pool_output_size * self.pool_output_size

        # 3. 權重參數一元化收納於 self.params 字典
        self.params = {}
        # Conv1 參數
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        
        # Affine1 參數
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_pixel, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        
        # Affine2 參數
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 4. 建立純淨無狀態算子實例 (初始化時不傳入 W 與 b)
        self.conv1 = Convolution(stride=filter_stride, padding=filter_pad)
        self.relu1 = ReluLayer()
        self.pool1 = Pooling(pool_h=pool_h, pool_w=pool_w, stride=pool_stride, padding=pool_pad, mode='max')
        
        self.affine1 = AffineLayer()
        self.relu2 = ReluLayer()
        self.affine2 = AffineLayer()
        
        self.last_layer = SoftmaxWithCeeLayer()

    def predict(self, x):
        """
        TODO 1: 實作無狀態算子前向傳播邏輯
        提示：
        1. 若輸入 x 形狀為 2D (N, 784)，自動 reshape 為 4D (N, 1, 28, 28)
        2. 呼叫 self.conv1.forward 時，需動態注入 self.params['W1'] 與 self.params['b1']
        3. 依序通過 conv1 -> relu1 -> pool1
        4. pool1 輸出為 4D，進入 affine1 前需展平 (Flatten) 為 2D 矩陣
        5. 呼叫 self.affine1.forward 時，動態注入 self.params['W2'] 與 self.params['b2']
        6. 依序通過 affine1 -> relu2 -> affine2 傳回分數 (scores)
        """
        # 1
        if x.ndim == 2:
            x = x.reshape(-1, 1, 28, 28)
        # 2, 3
        x = self.conv1.forward(x, self.params['W1'], self.params['b1'])
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        # 4 展平原本的四維張量形狀 (N, C, H, W)
        N, C, H, W = x.shape
        x_flat = x.reshape(N, -1)

        # 5, 6 
        x = self.affine1.forward(x_flat, self.params['W2'], self.params['b2'])
        x = self.relu2.forward(x)
        x = self.affine2.forward(x, self.params['W3'], self.params['b3'])

        return x


    def loss(self, x, t):
        score = self.predict(x)
        return self.last_layer.forward(score, t)

    def accuracy(self, x, t):
        score = self.predict(x)
        y = np.argmax(score, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

    def gradient(self, x, t):
        """
        TODO 2: 實作無狀態算子反向傳播與梯度收集邏輯
        提示：
        1. 呼叫 self.loss(x, t) 發動前向傳播並記錄中間快取
        2. 從 self.last_layer.backward(1.0) 啟動初始梯度
        3. 倒傳鏈：
           affine2.backward -> 獲得 dout, grads['W3'], grads['b3']
           relu2.backward   -> 獲得 dout
           affine1.backward -> 獲得 dout, grads['W2'], grads['b2']
        4. 利用 self.params['W1'].shape[0] 取得 FN，結合 self.pool_output_size 動態將 2D 梯度重構為 4D (N, FN, OH_pool, OW_pool)
        5. pool1.backward   -> 獲得 dout_pool
        6. relu1.backward   -> 獲得 dout_relu
        7. conv1.backward   -> 直接接收 dout_relu，攔截吐出的 tuple: _, grads['W1'], grads['b1']
        """
        # 1. 呼叫 self.loss(x, t) 發動前向傳播並記錄中間快取
        self.loss(x, t)
        grads = {}

        # 2. 從 self.last_layer.backward(1.0) 啟動初始梯度
        dout = 1.0
        dout = self.last_layer.backward(dout)

        #    3. 倒傳鏈：
        #    affine2.backward -> 獲得 dout, grads['W3'], grads['b3']
        #    relu2.backward   -> 獲得 dout
        #    affine1.backward -> 獲得 dout, grads['W2'], grads['b2']
        dout, grads['W3'], grads['b3'] = self.affine2.backward(dout)
        dout = self.relu2.backward(dout)
        dout, grads['W2'], grads['b2'] = self.affine1.backward(dout)


        # 4. 利用 self.params['W1'].shape[0] 取得 FN
        # 結合 self.pool_output_size 動態將 2D 梯度重構為 4D (N, FN, OH_pool, OW_pool)
        N = dout.shape[0]
        FN = self.params['W1'].shape[0]
        dout_4d = dout.reshape(N, FN, self.pool_output_size, self.pool_output_size)

        # 5. pool1.backward   -> 獲得 dout_pool
        dout_pool = self.pool1.backward(dout_4d)
        
        # 6. relu1.backward   -> 獲得 dout_relu
        dout_relu = self.relu1.backward(dout_pool)

        # 7. conv1.backward   -> 直接接收 dout_relu，攔截吐出的 tuple: _, grads['W1'], grads['b1']
        _, grads['W1'], grads['b1'] = self.conv1.backward(dout_relu)

        return grads

