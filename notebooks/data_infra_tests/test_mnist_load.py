# notebooks/test_mnist_load.py
from sklearn.datasets import fetch_openml

if __name__ == "__main__":
    print("正在從 OpenML 下載 MNIST 資料集（第一次下載需要花費一點時間）...")
    
    # fetch_openml 會全自動在專案根目錄建立 data/ 資料夾，並把數據鎖在裡面
    # version=1 是 MNIST 的標準版本，as_frame=True 代表我們明確要 Pandas 格式
    mnist = fetch_openml("mnist_784", version=1, data_home="data/", as_frame=True)
    
    # 提取特徵矩陣
    X = mnist.data
    # 提取真實標籤
    y = mnist.target
    
    print("\n 數據讀取成功！")
    print("-" * 50)
    print(f"X 的資料型態: {type(X)}")
    print(f"X的整體維度 (Shape): {X.shape}") # 預期應為 (70000, 784)
    print("-" * 50)
    
    # 直接利用 Pandas 的 head(10) 印出前 10 筆資料來觀察
    print("印出前 10 筆手寫圖片的像素矩陣 (部分欄位)：")
    print(X.head(10))
    
    print("\n印出前 10 筆資料對應的真實答案 (Target)：")
    print(y.head(10))

