import numpy as np
from ml_lab.layers import AffineLayer, ReluLayer, SoftmaxWithCeeLayer

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 1. 權重參數主權一元化：通通鎖死在 self.params 字典中
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
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
        # 1. 發動前向傳播，強迫各層內部節點動態記憶激活狀態與權重的矩陣肉體
        self.loss(x, t)
        
        # 2. 收集梯度的容器
        grads = {}
        
        # 3. 初始梯度發動：大合體終點節點輸出 (Y - T) / M
        dout = 1.0
        dout = self.last_layer.backward(dout) 
        
        # 4. 【請手動反向串接與精準攔截】
        # ===== 依序由後往前呼叫 backward，並當場攔截吐出的參數梯度 =====
        dout, grads['W2'], grads['b2'] = self.affine2.backward(dout)
        dout = self.relu1.backward(dout)
        _, grads['W1'], grads['b1'] = self.affine1.backward(dout)

        # =================================================================================
        return grads

