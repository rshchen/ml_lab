import numpy as np
import copy
from ml_lab.dataset import load_mnist
from ml_lab.models import TwoLayerNet
from ml_lab.optimizers import Adam
from ml_lab.hyperparameter import log_scale_sample


def run_coarse_search(x_train, t_train, x_val, t_val, num_trials=10, epochs=5, batch_size=128):
    """
    粗搜尋 (Coarse Search) 引擎
    """
    results = {}
    
    print("=" * 70)
    print(f" 啟動對數隨機超參數粗搜尋 (總試驗次數: {num_trials}, 單次 Epochs: {epochs})")
    print("=" * 70)

    for trial in range(num_trials):
        # ---------------------------------------------------------------------
        # 調用對數採樣算子取得學習率與權值衰減
        # lr 範圍: 10^-5 ~ 10^-1
        # weight_decay 範圍: 10^-8 ~ 10^-2
        # ---------------------------------------------------------------------

        lr = log_scale_sample(-5, -1)
        weight_decay = log_scale_sample(-8, -2)
        
        trial_key = f"Trial {trial+1:02d} | lr: {lr:.6f} | wd: {weight_decay:.8f}"
        print(f"\n>>> [{trial_key}]")
        
        # 建立網路與指定超參數的優化器
        net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
        optimizer = Adam(lr=lr, weight_decay=weight_decay)
        
        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size // batch_size, 1)
        
        val_acc_list = []

        for epoch in range(epochs):
            # 洗牌訓練集
            shuffle_mask = np.random.permutation(train_size)
            x_train_shuffled = x_train[shuffle_mask]
            t_train_shuffled = t_train[shuffle_mask]

            for i in range(iter_per_epoch):
                batch_mask = slice(i * batch_size, (i + 1) * batch_size)
                x_batch = x_train_shuffled[batch_mask]
                t_batch = t_train_shuffled[batch_mask]

                grads = net.gradient(x_batch, t_batch)
                optimizer.update(net.params, grads)

            # -----------------------------------------------------------------
            # 計算目前 Epoch 的訓練集與驗證集準確率
            # -----------------------------------------------------------------
            train_acc = net.accuracy(x_train, t_train)
            val_acc = net.accuracy(x_val, t_val)
            val_acc_list.append(val_acc)

            print(f"   Epoch {epoch+1:02d}/{epochs:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        results[trial_key] = {
            'lr': lr,
            'weight_decay': weight_decay,
            'val_acc_list': val_acc_list,
            'final_val_acc': val_acc_list[-1]
        }

    # -------------------------------------------------------------------------
    # [ Task 4 ] 根據最終驗證集準確率降冪排序搜尋結果
    # -------------------------------------------------------------------------
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_val_acc'], reverse=True)

    

    print("\n" + "=" * 70)
    print(" 粗搜尋完成！Top 3 最佳超參數組合如下：")
    print("=" * 70)
    for rank, (key, info) in enumerate(sorted_results[:3], 1):
        print(f" Rank {rank} | {key} | Final Val Acc: {info['final_val_acc']:.4f}")

    return sorted_results


if __name__ == "__main__":
    # 載入完整訓練集
    (x_train_full, t_train_full), (x_test, t_test) = load_mnist(normalize=True, one_hot=True)

    # -------------------------------------------------------------------------
    # 切分 Validation Set (取訓練集前 20% 即 12,000 筆作為驗證集)
    # 嚴禁使用 test set 來挑選超參數！
    # -------------------------------------------------------------------------
    val_size = int(x_train_full.shape[0] * 0.2)
    
    x_val = x_train_full[:val_size]
    t_val = t_train_full[:val_size]
    x_train = x_train_full[val_size:]
    t_train = t_train_full[val_size:]

    # 執行 5 次試驗的粗搜尋驗證
    run_coarse_search(x_train, t_train, x_val, t_val, num_trials=5, epochs=3)

