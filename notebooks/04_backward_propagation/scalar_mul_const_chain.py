import numpy as np

class ScalarMulConstLayer:
    """
    計算圖拓撲中的單變數常數乘法算子
    核心映射：y = x * const_val
    """
    def __init__(self, const_val):
        self.const_val = const_val
        self.x = None

    def forward(self, x):
        # 記錄前向傳播狀態至快取，以備反向求導調用
        self.x = x
        out = x * self.const_val
        return out

    def backward(self, dout):
        # 根據單變數乘法函數 f(x) = x * c 的導數公式：f'(x) = c
        # 調用快取狀態，將上游梯度 dout 乘以局部導數，計算下游梯度
        dx = dout * self.const_val
        return dx

# =====================================================================
# 拓撲鏈條串聯與數值驗證
# =====================================================================

# 1. 建立拓撲節點物件，並在宣告時直接注入常數環境（單價與稅率）
apple_price_layer = ScalarMulConstLayer(const_val=100)
tax_layer = ScalarMulConstLayer(const_val=1.1)

# 2. 定義初始唯一自變數
apple_num = 2

# 3. 前向傳播（Forward Pass）由左往右計算單變數複合鏈條
inter_apple_price = apple_price_layer.forward(apple_num)
total_loss = tax_layer.forward(inter_apple_price)

print(f"[Forward] 中間層總價 (h): {inter_apple_price}") # 預期：200
print(f"[Forward] 最終總花費 (L): {total_loss}")        # 預期：220

# 4. 反向傳播（Backward Pass）給定初始梯度由右往左計算
d_initial_loss = 1.0

# 5. 逐步回溯節點，計算全域偏導數
d_inter_price = tax_layer.backward(d_initial_loss)
d_apple_num = apple_price_layer.backward(d_inter_price)

print(f"[Backward] 損失對中間層總價的偏導數 (dL/dh): {d_inter_price}") # 預期：1.1
print(f"[Backward] 損失對蘋果數量的偏導數 (dL/dx): {d_apple_num}")     # 預期：110.0

# 數理正當性檢查
assert np.isclose(total_loss, 220.0)
assert np.isclose(d_apple_num, 110.0)
print("\n[Congratulation] 拓撲計算圖與 Salas 單變數鏈鎖律數值完全對齊，通車成功！")

