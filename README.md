# 介紹
這是一個 machine learning 的自學專案。
# 虛擬環境
虛擬環境以 `uv` 作為管理工具。
# 參考資料
## 書籍
- Deep Learning｜用 Python 進行深度學習的基礎理論實作/斎藤康毅
# 檔案結構
- 常會用到的函數整理到了 `src/ml_lab` 當中。
- notebooks 是練習內容，根據魚書的內容和我認為重要的核心訓練整理如下：
    - `perceptron.py` 實作了感知器 `and_gate, or_gate, nand_gate`，從數學上我們知道單層感知器只能得到線性分類器，但利用 `and, or, nand` 我們可以組合出多層感知器，實現非線性分類器，例如 `xor_gate`。
    - `activation_function.py` 隱藏層激勵函數實作，包含 matploblib 繪圖練習。
    - `softmax_bar.py` 輸出層激勵函數 softmax 實作，包含統計圖練習。
    - `mlp_two_layer_propagation.py` 向前傳播矩陣維度對其的簡易練習。
    - `test_mnist_load.py` 使用 SciKit 下載 MNIST 資料集。
    - `test_mnist_visualize.py` 使用 matplotlib 將 MNIST 資料集視覺化。
    - `test_pickle_craft.py` 練習自己 dump `.pkl` 並 load 回來使用。
    - `inspect_pkl.py` 拿到一個 `.pkl`，如何分析檔案內容的方法。
    - `mlp_mnist_inference.py` 利用魚書的 `.pkl` 獲得權重，完成正向傳播。
    - `mlp_mnist_inference_batch.py` 利用魚書的 `.pkl` 獲得權重，完成「批次」正向傳播。
