# 介紹
這是一個 machine learning 的自學專案。
# 虛擬環境
虛擬環境以 `uv` 作為管理工具。
# 參考資料
## 書籍
- 斎藤康毅的 Deep Learning 系列（又稱魚書）。
    - Deep Learning｜用 Python 進行深度學習的基礎理論實作/斎藤康毅

這是一份為你更新後的專案結構與模組導航筆記。

這次更新不僅精準對齊了你終端機輸出的最新 `tree` 結構（包含 `openml` 的完整層級、`__pycache__` 编译快取，以及你在第四章後半段大刀闊斧新增與重構的優化核心組件），也保留了原本嚴謹的工程筆記風格。

---

## 檔案結構與模組導航

本專案全面採用 `uv` 進行依賴管理，核心代碼邏輯拆分為高度解耦的 `src/` 核心庫與 `notebooks/` 漸進式實驗室：

```text
.
├── README.md
├── pyproject.toml              # 專案依賴與環境配置定稿
├── uv.lock                     # uv 鎖定的精準重現環境檔案
├── main.py                     # 專案主入口調用腳本
├── data/                       # 持久化資料集與權重模型快取
│   ├── sample_weight.pkl       # 魚書官方提供的預訓練權重
│   ├── my_test_weight.pkl      # 練習自己 dump 產出的實驗權重
│   └── openml/                 # SciKit-Learn 自動下載並解壓的 MNIST 原始數據物理快取
│       └── openml.org/
│           ├── api/v1/json/data/... # 詮釋資料與特徵屬性的 JSON 快取檔
│           └── data/v1/download/... # 核心圖像資料集樣本 (mnist_784.arff.gz)
├── src/ml_lab/                 # 核心公共底層函數庫 (高內聚、解耦框架核心)
│   ├── __init__.py
│   ├── activations.py          # 封裝所有隱藏層與輸出層的激活函數 (如 Sigmoid, Softmax)
│   ├── losses.py               # 封裝多種損失函數 (如 MSE, Cross Entropy Error)
│   ├── gradient.py             # 【新增】通用數值微分算子專區 (含通用多維張量中心差分)
│   ├── optimizer.py            # 【新增】優化器演算法引擎核心 (含基礎梯度下降法梯度下降法 SGD)
│   └── utils.py                # 封裝資料集讀取、環境路徑初始化等工具函數
└── notebooks/                  # 技術專題階段性練習與視覺化實驗室

```

---

### notebooks 子模組細節清單

#### 1. `01_perceptrons/` — 感知器基石

* **`perceptron.py`**：實作感知器 `and_gate`, `or_gate`, `nand_gate`。單層感知器在數學上僅能建構線性分類器，但透過多層感知器（MLP）的邏輯組合，成功攻克非線性分類的經典 `xor_gate` 難題。

#### 2. `02_feedforward_inference/` — 前向傳播與多維推理

* **`activation_function.py`**：隱藏層激勵函數的實作與代數邊界確認，包含 Matplotlib 曲線繪製練習。
* **`softmax_bar.py`**：輸出層激勵函數 Softmax 的溢出防御實作與統計圖表練習。
* **`mlp_two_layer_propagation.py`**：簡易的多層矩陣點積乘法練習，用於打通矩陣維度對齊（Shape Matching）的直覺。
* **`mlp_mnist_inference.py`**：利用 `data/` 的官方預訓練 `.pkl` 權重，完成單張 MNIST 圖像的前向傳播推理。
* **`mlp_mnist_inference_batch.py`**：前向傳播的矩陣化進階版，導入「批次處理（Batch Inference）」機制，大幅優化計算機記憶體吞吐效率。

#### 3. `03_optimization_losses/` — 損失函數、微積分與最佳化軌跡

* **`test_losses_naive.py`**：對單一樣本計算 Mean Squared Error (MSE) 與 Cross Entropy Error (CEE) 的初始嘗試。
* **`test_losses_batch.py`**：將損失函數矩陣化，支援批次樣本 Loss 的平行化計算（對齊 `src/ml_lab/losses.py`）。
* **`test_numerical_diff.py`**：數值微分的中心差分實作，探討浮點數下溢與機器捨入誤差的硬體妥協。
* **`visualize_3d_bowl.py`**：繪製 $f(x_0, x_1) = x_0^2 + x_1^2$ 的 3D 碗狀 Loss 曲面，建立切平面逼近的立體空間直覺。
* **`visualize_gradient_field.py`**：利用 `meshgrid`、`contour` 與 `quiver` 繪製二維權重底面的等高線與反梯度方向箭頭場。
* **`gradient_descent_core.py`**：【新增】優化核心測試，驗證通用多維數值微分算子在二維座標系下的收斂正確性。
* **`gradient_simplenet.py`**：【新增】參數空間映射實驗，利用高階函數閉包（Lambda）將神經網路的權重矩陣 $W$ 參數化為自變數，完成單層網路的數值梯度概念驗證（PoC）。
* **`plot_lr_topology.py`**：【新增】步長拓撲學實驗，利用 `plt.subplots` 橫向對比不同 $\eta$（學習率）下的離散時間更新軌跡，視覺化呈現發散（Over-shooting）與收斂過慢（Under-shooting）的地貌。

#### 4. `data_infra_tests/` — 資料基礎設施與環境測試

* **`test_for_root_dir.py`**：驗證跨目錄調用時，相對路徑與專案根目錄（Root Directory）的解析合法性，確保資料讀取不卡死。
* **`test_mnist_load.py`**：使用 SciKit-Learn 下載、解壓並快取 `mnist_784` 原始資料集。
* **`test_mnist_visualize.py`**：使用 Matplotlib 將離散的一維 $1 \times 784$ 陣列重塑（`reshape`）為 $28 \times 28$ 像素矩陣並進行影像視覺化。
* **`test_pickle_craft.py`**：二進位序列化練習，親手完成權重字典的 `pickle.dump` 與 `load` 封裝。
* **`inspect_pkl.py`**：面對未知黑盒子 `.pkl` 檔案時，主動分析其內部字典鍵值（Keys）、張量形狀（Shapes）與資料型態的方法學。