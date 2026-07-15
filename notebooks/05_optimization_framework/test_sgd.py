import numpy as np
from ml_lab.dataset import load_mnist
from ml_lab.models import TwoLayerNet
from ml_lab.optimizers import Sgd 

if __name__ == "__main__":
    (x_train, t_train), (_, _) = load_mnist(normalize=True, one_hot=True)
    
    batch_size = 100
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    optimizer = Sgd(lr=0.1)
    
    print("=== 啟動 Sgd 優化器組裝測試 ===")
    for i in range(3):
        batch_mask = np.random.choice(x_train.shape[0], batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        grads = network.gradient(x_batch, t_batch)
        loss_before = network.loss(x_batch, t_batch)
        
        # ----------------------------------------------------------------
        # 呼叫優化器實作參數主權一元化更新
        # ----------------------------------------------------------------
        optimizer.update(network.params, grads)

        loss_after = network.loss(x_batch, t_batch)
        print(f"迭代 {i+1} | 更新前 Loss: {loss_before:.4f} -> 更新後 Loss: {loss_after:.4f}")

