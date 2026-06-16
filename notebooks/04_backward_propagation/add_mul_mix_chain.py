import numpy as np

class ScalarMulConstLayer:
    """
    計算圖拓撲中的單變數常數乘法算子
    """
    def __init__(self, const_val):
        self.const_val = const_val

    def forward(self, x):
        out = x * self.const_val
        return out

    def backward(self, dout):
        dx = dout * self.const_val
        return dx

class AddLayer:
    """
    計算圖拓撲中的二元加法算子
    核心映射：z = x + y
    """
    def __init__(self):
        # 加法算子不持有狀態快取
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        # ---------------------------------------------------------
        # 根據加法函數 z = x + y 的偏導數公式：∂z/∂x = 1, ∂z/∂y = 1
        # 將上游梯度 dout 乘以局部導數，分別計算流向兩個輸入端的下游梯度
        # ---------------------------------------------------------
        dx = dout * 1
        dy = dout * 1
        # ---------------------------------------------------------
        return dx, dy

# =====================================================================
# 雙通道拓撲鏈條串聯與數值驗證
# =====================================================================

# 1. 建立拓撲節點物件
apple_price_layer = ScalarMulConstLayer(const_val=100)
orange_price_layer = ScalarMulConstLayer(const_val=150)
fruit_add_layer = AddLayer()
tax_layer = ScalarMulConstLayer(const_val=1.1)

# 2. 定義初始獨立自變數環境
apple_num = 2
orange_num = 3

# 3. 前向傳播（Forward Pass）由左往右進行平行計算與匯聚
# (1) 蘋果通道與橘子通道獨立平行開工
inter_apple_price = apple_price_layer.forward(apple_num)
inter_orange_price = orange_price_layer.forward(orange_num)

# (2) 在加法節點匯聚
total_fruit_price = fruit_add_layer.forward(inter_apple_price, inter_orange_price)

# (3) 注入消費稅層得到最終損失
total_loss = tax_layer.forward(total_fruit_price)

print(f"[Forward] 蘋果中間層總價 (h1): {inter_apple_price}") # 預期：200
print(f"[Forward] 橘子中間層總價 (h2): {inter_orange_price}") # 預期：450
print(f"[Forward] 水果匯聚總總價 (h3): {total_fruit_price}")  # 預期：650
print(f"[Forward] 最終總花費 (L): {total_loss}")             # 預期：715

# 4. 反向傳播（Backward Pass）給定初始梯度由右往左進行逆向倒灌
d_initial_loss = 1.0

# 5. 逐步回溯節點，計算各通道全域偏導數
# (1) 回溯稅率層
d_total_fruit_price = tax_layer.backward(d_initial_loss)

# (2) 回溯加法節點，執行梯度解耦分流
d_inter_apple_price, d_inter_orange_price = fruit_add_layer.backward(d_total_fruit_price)

# (3) 平行回溯兩側單價算子，收割底層自變數完全體梯度
d_apple_num = apple_price_layer.backward(d_inter_apple_price)
d_orange_num = orange_price_layer.backward(d_inter_orange_price)

print(f"\n[Backward] 損失對水果匯聚總價的偏導數 (dL/dh3): {d_total_fruit_price}")  # 預期：1.1
print(f"[Backward] 損失對蘋果中間總價的偏導數 (dL/dh1): {d_inter_apple_price}") # 預期：1.1
print(f"[Backward] 損失對橘子中間總價的偏導數 (dL/dh2): {d_inter_orange_price}") # 預期：1.1
print(f"[Backward] 損失對蘋果數量的偏導數 (dL/dx1): {d_apple_num}")            # 預期：110.0
print(f"[Backward] 損失對橘子數量的偏導數 (dL/dx2): {d_orange_num}")            # 預期：165.0

# 數理正當性斷言檢查
assert np.isclose(total_loss, 715.0)
assert np.isclose(d_apple_num, 110.0)
assert np.isclose(d_orange_num, 165.0)
print("\n[Congratulation] 雙通道複合計算圖與 Salas 多元偏微分鏈鎖律數值大對齊，通車成功！")

