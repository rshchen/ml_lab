import numpy as np
from ml_lab.dataset import load_mnist
from ml_lab.layers import BatchNormalization

if __name__ == "__main__":
    np.random.seed(42)
    (x_train, _), (_, _) = load_mnist(normalize=True, one_hot=True)
    
    batch_size = 100
    feature_dim = x_train.shape[1] # MNIST 為 784
    
    # 1. 初始化 BN 可學習參數 (gamma 設為 1, beta 設為 0)
    gamma = np.ones(feature_dim)
    beta = np.zeros(feature_dim)
    bn_layer = BatchNormalization(gamma, beta, momentum=0.9, eps=1e-5)
    
    print("=== 啟動 BatchNormalization 模組組裝測試 ===")
    
    # ----------------------------------------------------------------
    # 測試階段 1：訓練模式 (Training Phase) 與 EMA 更新
    # ----------------------------------------------------------------
    print("\n--- [階段 1: 訓練模式前向與反向傳播] ---")
    for i in range(3):
        batch_mask = np.random.choice(x_train.shape[0], batch_size)
        x_batch = x_train[batch_mask]
        
        # 訓練模式前向傳播
        out_train = bn_layer.forward(x_batch, is_train=True)
        
        batch_mean = np.mean(out_train)
        batch_var = np.var(out_train)
        
        print(f"迭代 {i+1} | 輸出 Batch 均值: {batch_mean:.6f} (目標 ≈ 0) | 輸出 Batch 方差: {batch_var:.6f} (目標 ≈ 1)")
        
        # 模擬來自後層的反向傳播梯度
        dout = np.random.randn(*out_train.shape)
        dx, dgamma, dbeta = bn_layer.backward(dout)
        
        assert dx.shape == x_batch.shape, "dx 形狀與輸入不一致！"
        assert dgamma.shape == gamma.shape, "dgamma 形狀與 gamma 不一致！"
        assert dbeta.shape == beta.shape, "dbeta 形狀與 beta 不一致！"

    print("\n反向傳播梯度維度檢查完全通過！")
    print(f"累積之 Running Mean 模長: {np.linalg.norm(bn_layer.running_mean):.4f}")
    print(f"累積之 Running Var 模長: {np.linalg.norm(bn_layer.running_var):.4f}")

    # ----------------------------------------------------------------
    # 測試階段 2：推論模式 (Inference Phase) 評估
    # ----------------------------------------------------------------
    print("\n--- [階段 2: 推論模式 (is_train=False)] ---")
    # 抽取單一測試樣本 (1, 784)
    x_test_single = x_train[0:1] 
    out_test = bn_layer.forward(x_test_single, is_train=False)
    
    print(f"推論模式單一樣本輸出形狀: {out_test.shape}")
    print(f"推論模式輸出 L2 範數: {np.linalg.norm(out_test):.4f}")
    print("\n=== BatchNormalization 模組測試全部成功結束 ===")