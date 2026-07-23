import numpy as np
from ml_lab.layers import AffineLayer, ReluLayer, SoftmaxWithCeeLayer

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

