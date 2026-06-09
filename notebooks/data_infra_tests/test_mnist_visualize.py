# notebooks/test_mnist_visualize.py
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

if __name__ == "__main__":
    print("正在從本地 data/ 讀取 MNIST 數據...")
    
    mnist = fetch_openml("mnist_784", version=1, data_home="data/", as_frame=True)
    
    X = mnist.data.head(100)
    y = mnist.target.head(100)
    
    # 挑選第 0 筆資料進行視覺化
    sample_index = 1
    
    # 提取第 0 筆的 784 個像素，並轉換成 numpy array
    single_image_1d = X.iloc[sample_index].to_numpy()
    correct_label = y.iloc[sample_index]
    
    # 請填空：將 1D 向量變回 2D 畫布
    single_image_2d = single_image_1d.reshape(28, 28)
    
    # 請填空：使用 matplotlib 進行繪圖，設定正確的灰階參數
    plt.imshow(single_image_2d, cmap="gray")
    
    # 設定標題並關閉刻度軸
    plt.title(f"Label (Answer) is: {correct_label}")
    plt.axis("off")
    
    print(f"正在彈出第 {sample_index} 筆資料的圖片視窗...")
    plt.show()

