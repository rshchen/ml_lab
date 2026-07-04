import numpy as np
from ml_lab.dataset import load_mnist  # 使用自製的數據讀取工具
from ml_lab.models import TwoLayerNet  # 引入你剛完成的顯式拓撲模型

if __name__ == "__main__":
    # 讀取 MNIST 數據 (訓練集 60,000 筆，測試集 10,000 筆)
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot=True)
    
    # 初始化超參數
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    
    # 宣告 784 50 10 的顯式拓撲計算圖網路
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    # 追蹤指標儲存庫
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    # 計算每個 Epoch 的迭代次數 (60000 / 100 = 600 步)
    iter_per_epoch = max(train_size // batch_size, 1)
    
    print("=== 計算圖引擎啟動：MNIST 終極參數進化 ===")
    for i in range(iters_num):
        # 1.【Mini-batch 數據抽樣】
        # ===== 隨機抽選出 batch_size 筆資料，並切片出 x_batch 與 t_batch =====
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        # =================================================================================
        
        # 2. 【解析梯度獲取】
        # ===== 呼叫計算圖顯式反向傳播，取得當前批次的解析梯度字典 =====
        grads = network.gradient(x_batch, t_batch)
        # =================================================================================
        
        # 3. 【參數主權一元化更新】
        # ===== 依據學習率與梯度，直接原地優化網路內部的 params 變數 =====
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grads[key]
        
        # =================================================================================
            
        # 記錄當前批次的損失值
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        
        # 4. 【分流指標觀測】每隔一個 Epoch，進行一次「訓練集 vs 測試集」的雙線精準度大對帳
        # ===== 在 Epoch 節點結算 Accuracy 並 append 存入指標儲存庫 =====
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"Epoch {(i // iter_per_epoch) + 1} | Loss: {loss:.4f} | Train Acc: {train_acc * 100:.2f}% | Test Acc: {test_acc * 100:.2f}%")

