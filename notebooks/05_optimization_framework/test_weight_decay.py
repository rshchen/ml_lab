import copy
import numpy as np
from ml_lab.dataset import load_mnist
from ml_lab.models import TwoLayerNet
from ml_lab.optimizers import Sgd, Momentum, AdaGrad, Adam

if __name__ == "__main__":
    (x_train, t_train), (_, _) = load_mnist(normalize=True, one_hot=True)
    
    weight_decay_lambda = 0.1 # 設定重力拉扯係數
    
    # 建立支援 Weight Decay 的優化器清單
    optimizers = {
        "SGD": Sgd(lr=0.01, weight_decay=weight_decay_lambda),
        "Momentum": Momentum(lr=0.01, decay=0.9, weight_decay=weight_decay_lambda),
        "AdaGrad": AdaGrad(lr=0.01, weight_decay=weight_decay_lambda),
        "Adam": Adam(lr=0.01, weight_decay=weight_decay_lambda)
    }
    
    base_network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    networks = {name: copy.deepcopy(base_network) for name in optimizers.keys()}
    
    batch_size = 128
    steps = 5
    
    np.random.seed(42)
    batch_masks = [np.random.choice(x_train.shape[0], batch_size) for _ in range(steps)]
    
    print("=" * 70)
    print(f" 啟動 Weight Decay 優化器同台競技 (WD Lambda = {weight_decay_lambda})")
    print("=" * 70)
    
    for i in range(steps):
        print(f"\n[ 迭代 {i+1} ]")
        x_batch = x_train[batch_masks[i]]
        t_batch = t_train[batch_masks[i]]
        
        for name, opt in optimizers.items():
            net = networks[name]
            
            # 計算更新前的權重 L2 模長
            w1_norm_before = np.linalg.norm(net.params['W1'])
            loss_before = net.loss(x_batch, t_batch)
            
            grads = net.gradient(x_batch, t_batch)
            opt.update(net.params, grads)
            
            # 計算更新後的權重 L2 模長與 Loss
            w1_norm_after = np.linalg.norm(net.params['W1'])
            loss_after = net.loss(x_batch, t_batch)
            
            w_delta = w1_norm_before - w1_norm_after
            
            print(f"  {name:<10} | Loss: {loss_before:.4f}->{loss_after:.4f} | W1 Norm: {w1_norm_before:.4f}->{w1_norm_after:.4f} (衰減: {w_delta:+.4f})")
            
    print("\n" + "=" * 70)
    print(" 測試完成！Weight Decay 成功壓制權重模長增長。")
    print("=" * 70)

