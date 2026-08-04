import numpy as np
import time
from ml_lab.dataset import load_mnist
from ml_lab.models import SimpleConvNet
from ml_lab.optimizers import Adam

# 載入數據
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot=True)

# 取前 5,000 筆訓練資料與 1,000 筆測試資料進行快速收斂驗證
x_train_small = x_train[:5000]
t_train_small = t_train[:5000]
x_test_small = x_test[:1000]
t_test_small = t_test[:1000]

# 實作模型與優化器
network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 3, 'pad': 1, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

optimizer = Adam(lr=0.001)

# 超參數設定
iters_num = 300
batch_size = 100
train_size = x_train_small.shape[0]

print("=== 開始 MNIST 端到端無狀態算子收斂測試 ===")
start_time = time.time()

for i in range(iters_num):
    # 隨機採樣 Mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train_small[batch_mask]
    t_batch = t_train_small[batch_mask]

    # 計算梯度與更新參數 (無狀態架構自動完成參數對齊)
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if (i + 1) % 50 == 0 or i == 0:
        loss = network.loss(x_batch, t_batch)
        acc = network.accuracy(x_train_small[:500], t_train_small[:500])
        print(f"Step {i+1:3d} / {iters_num} | Loss: {loss:.4f} | Train Acc: {acc * 100:.2f}%")

end_time = time.time()
test_acc = network.accuracy(x_test_small, t_test_small)

print(f"\n=== 測試完成 (耗時: {end_time - start_time:.2f} 秒) ===")
print(f"最終 Test Accuracy (1000 samples): {test_acc * 100:.2f}%")

