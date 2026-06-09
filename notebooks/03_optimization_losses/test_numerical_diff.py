import numpy as np

def target_function(x):
    """測試函數: f(x) = x^3"""
    return x ** 3

def analytical_derivative(x):
    """微積分解析解 (真實導數): f'(x) = 3x^2"""
    return 3 * (x ** 2)

def forward_difference(f, x, h):
    """前向差分公式"""
    return (f(x + h) - f(x))/h

def central_difference(f, x, h):
    """中心差分公式 """
    return (f(x + h) - f(x - h))/(2 * h)

if __name__ == "__main__":
    print("==================================================")
    print("數值微分精確度與計算機地雷實驗")
    print("==================================================")
    
    x_test = 2.0
    true_ans = analytical_derivative(x_test) # 3 * (2^2) = 12.0
    print(f"在 x = {x_test} 處的數學真實導數 (解析解) = {true_ans}\n")
    
    # --------------------------------------------------
    # 實驗一：驗證計算機的「捨入誤差地雷」
    # --------------------------------------------------
    print("【實驗一：當 h 盲目追求極小會發生什麼事？】")
    print("-" * 50)
    h_disaster = 1e-20  # 超越硬體極限的步伐
    
    err_forward_disaster = forward_difference(target_function, x_test, h_disaster)
    print(f"h = 1e-20 時的前向差分結果 : {err_forward_disaster}")
    print("  (原因：h 太小導致浮點數下溢與相消誤差，計算機直接吐出亂碼或 0)")
    print("-" * 50 + "\n")
    
    # --------------------------------------------------
    # 實驗二：驗證 $O(h)$ vs $O(h^2)$ 的萬倍差距
    # --------------------------------------------------
    print("【實驗二：在黃金步長 h = 1e-4 下，兩種差分的精確度對決】")
    print("-" * 50)
    h_golden = 1e-4
    
    ans_forward = forward_difference(target_function, x_test, h_golden)
    ans_central = central_difference(target_function, x_test, h_golden)
    
    # 計算絕對誤差
    diff_forward = abs(ans_forward - true_ans)
    diff_central = abs(ans_central - true_ans)
    
    print(f"前向差分計算結果 : {ans_forward:<15} | 絕對誤差 : {diff_forward:.10f}")
    print(f"中心差分計算結果 : {ans_central:<15} | 絕對誤差 : {diff_central:.10f}")
    
    print(f"\n 實驗勝負報告：")
    print(f" 中心差分的精確度大約是前向差分的 {int(diff_forward / diff_central)} 倍！")
    print(f" 這完美印證了泰勒展開式中 O(h) 與 O(h^2) 的數學證明。")
    print("-" * 50)

