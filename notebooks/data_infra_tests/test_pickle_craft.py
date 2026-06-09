# notebooks/test_pickle_craft.py
import pickle
import numpy as np

if __name__ == "__main__":
    print("--- 階段一：親手製作你的參數罐頭 ---")
    
    # 模擬一組微調好的神經網路參數
    my_network = {
        "W1": np.array([[0.1, 0.2], [0.3, 0.4]]),
        "b1": np.array([0.5, 0.6]),
        "layer_name": "Hidden_Layer_1"
    }
    
    # 1. 請填空：使用正確的二進位「寫入」模式，將字典打包存進硬碟
    with open("data/my_test_weight.pkl", "wb") as f:
        pickle.dump(my_network, f)
        
    print("成功將參數矩陣醃製成 data/my_test_weight.pkl 檔案！")
    print("-" * 50)
    
    # 強制清空記憶體中的變數，確保待會讀出來的不是舊資料
    del my_network
    
    print("--- 階段二：開箱解鎖你的參數罐頭 ---")
    
    # 2. 請填空：使用正確的二進位「讀取」模式，將硬碟中的資料還原到記憶體
    with open("data/my_test_weight.pkl", "rb") as f:
        loaded_network = pickle.load(f)
        
    # 3. 驗證還原出來的數據與型態
    print("解開後的欄位名稱：", loaded_network.keys())
    print("W1 矩陣內容：\n", loaded_network["W1"])
    print("W1 的型態（是否依然是熟悉的 NumPy 陣列？）：", type(loaded_network["W1"]))
    print("網路層名稱：", loaded_network["layer_name"])
    
    print("-" * 50)
    print("驗證完畢！你已經完全掌握了 Pickle 的製作與讀取核心技術！")

