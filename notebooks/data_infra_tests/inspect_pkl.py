# notebooks/inspect_pkl.py
import pickle

if __name__ == "__main__":
    file_path = "data/sample_weight.pkl"
    
    print(f"正在對 {file_path} 進行黑盒子結構掃描...\n")
    
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        print("======== 檔案外殼資訊 ========")
        print(f"最外層資料型態 (Type): {type(data)}")
        
        if isinstance(data, dict):
            print(f"包含的 Key 數量: {len(data)}")
            print(f"欄位清單 (Keys): {list(data.keys())}\n")
            
            print("======== 內部矩陣維度報告 ========")
            for k, v in data.items():
                # 檢查裡面是不是 numpy 陣列
                if hasattr(v, "shape"):
                    print(f"參數 [{k:4}] -> 型態: NumPy Array | Shape: {v.shape} | 資料型態: {v.dtype}")
                else:
                    print(f"參數 [{k:4}] -> 型態: {type(v)} | 內容: {v}")
                    
        else:
            print("這不是一個字典包裹，它是一個獨立的物件：")
            print(data)
            
        print("====================================")
        print("掃描完成！你已經看穿了這個 pkl 的內部長相。")
        
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 {file_path}，請確認檔案已放置在對應路徑。")

