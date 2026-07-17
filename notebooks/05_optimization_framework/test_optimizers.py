import copy
import numpy as np
from ml_lab.dataset import load_mnist
from ml_lab.models import TwoLayerNet
# 隨著你開發進度，直接在這裡 import 新優化器
from ml_lab.optimizers import Sgd, Momentum, AdaGrad, Adam

if __name__ == "__main__":
    # 1. 載入 MNIST
    (x_train, t_train), (_, _) = load_mnist(normalize=True, one_hot=True)
    
    # 2. 定義要測試與對比的優化器清單
    # 未來寫好 AdaGrad 和 Adam 只要取消註解即可，完全不需改動下方測試邏輯
    optimizers = {
        "SGD": Sgd(lr=0.01),
        "Momentum": Momentum(lr=0.01, decay=0.9),
        "AdaGrad": AdaGrad(lr=0.01),
        "Adam": Adam(lr=0.001)
    }
    
    # 3. 建立基礎網路
    base_network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    # 4. 為每個優化器生成完全一致的「平行宇宙」網路副本
    networks = {name: copy.deepcopy(base_network) for name in optimizers.keys()}
    
    batch_size = 128
    steps = 5
    
    # 固定隨機數，確保每一步所有優化器拿到的 Mini-batch 完完全全相同
    np.random.seed(42)
    batch_masks = [np.random.choice(x_train.shape[0], batch_size) for _ in range(steps)]
    
    print("=" * 60)
    print(f" 啟動多優化器同台競技 (測試步數: {steps})")
    print("=" * 60)
    
    for i in range(steps):
        print(f"\n[ 迭代 {i+1} ]")
        x_batch = x_train[batch_masks[i]]
        t_batch = t_train[batch_masks[i]]
        
        for name, opt in optimizers.items():
            net = networks[name]
            
            # 計算目前 Loss 與梯度
            loss_before = net.loss(x_batch, t_batch)
            grads = net.gradient(x_batch, t_batch)
            
            # 更新參數
            opt.update(net.params, grads)
            
            # 計算更新後 Loss
            loss_after = net.loss(x_batch, t_batch)
            delta = loss_before - loss_after
            
            print(f"  {name:<10} | Loss: {loss_before:.4f} -> {loss_after:.4f} (降幅: {delta:.4f})")
            
    print("\n" + "=" * 60)
    print(" 測試完成！所有優化器狀態與參數皆成功更新。")
    print("=" * 60)

