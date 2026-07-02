import pickle
import numpy as np
from pathlib import Path
from sklearn.datasets import fetch_openml

# 利用 pathlib 自適應 src-layout 結構，向外跳三階回到專案根目錄
ROOT_DIR   = Path(__file__).resolve().parents[2]
CACHE_DIR  = ROOT_DIR / "data"
CACHE_FILE = CACHE_DIR / "mnist.pkl"

def load_mnist(normalize=True, one_hot=True):
    """
    自製 MNIST 資料加載管道：首抽下載至 data/，永久本地秒級加載
    """
    # 確保 data/ 資料夾存在
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
    # 1. 【第一道防線：本地二進位快取檢查】
    if CACHE_FILE.exists():
        # ===== 利用 pathlib 與 pickle 從本地秒讀取快取字典 =====
        with CACHE_FILE.open("rb") as f:
            dataset = pickle.load(f)
        
        # ===================================================================
        print("=== [Cache] 成功從本地秒速載入 MNIST 原始檔案 ===")
    else:
        print("=== [OpenML] 本地無快取，正在線上拉取 MNIST 數據 (首次執行需耗時) ===")
        # ===== 呼叫 fetch_openml，將下載點指向 CACHE_DIR 並強行封鎖 Pandas =====
        mnist = fetch_openml('mnist_784', version=1, data_home=str(CACHE_DIR), as_frame=False, parser='auto')

        # ===================================================================================
        
        # 抽取原始數據
        dataset = {
            'X': mnist.data,
            'y': mnist.target.astype(int) # 強制將字串 '0'~'9' 轉為整數 0~9
        }
        
        # ===== 將 dataset 字典原地「醃製」存檔至 CACHE_FILE =====
        with CACHE_FILE.open('wb') as f:
            pickle.dump(dataset, f)
        
        
        # ======================================================================
        print(f"=== [Cache] 原始數據已成功醃製存檔至: {CACHE_FILE} ===")

    X, y = dataset['X'], dataset['y']

    # 2. 【第二道防線：動態歸一化地貌】
    if normalize:
        # ===== 將 0~255 的像素投射至 0.0~1.0 幾何區間 =====
        X = (X / 255.0).astype(np.float32)
        # =============================================================

    # 3. 【第三道防線：進階二維索引無迴圈 One-Hot 化】
    if one_hot:
        num_classes = 10
        # ===== 利用進階索引，在不發動 for 迴圈下完成 One-Hot 矩陣化 =====
        y_one_hot = np.zeros((y.shape[0], num_classes))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        y = y_one_hot
        # ===========================================================================

    # 4. 標準 60000 / 10000 幾何切片通車
    x_train, x_test = X[:60000], X[60000:]
    t_train, t_test = y[:60000], y[60000:]

    return (x_train, t_train), (x_test, t_test)

