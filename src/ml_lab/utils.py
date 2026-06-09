import pickle
from pathlib import Path

def load_network(file_name: str = "sample_weight.pkl"):
    """
    載入指定名稱的神經網路權重與偏置 (預設為魚書的 sample_weight.pkl)
    運用 Path 確保不論在哪個目錄執行，皆能精準對齊專案根目錄的 data/ 資料夾
    """
    # 1. 全自動定位到當前專案的根目錄
    project_root = Path(__file__).resolve().parents[2]
    
    # 2. 動態拼接路徑：將預設或傳入的 file_name 組合進 data/ 資料夾
    weight_path = project_root / "data" / file_name
    
    if not weight_path.exists():
        raise FileNotFoundError(f"找不到權重檔案，請確認檔案是否存在於: {weight_path}")
        
    with open(weight_path, "rb") as f:
        network = pickle.load(f)
        
    return network

def numerical_diff(f, x, h=1e-4):
    """
    通用單變數數值微分 (中心差分)
    預設使用黃金步長 h = 1e-4，確保在雙精度浮點數下壓制截斷誤差至 O(h^2)
    """
    return (f(x + h) - f(x - h)) / (2 * h)