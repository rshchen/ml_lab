import numpy as np
class BinaryAddLayer:
    def __init__(self):
        pass
    def forward(self, x, y):
        out  = x + y
        return out
    def backward(self, dout):
        dx = dout
        dy = dout
        return dx, dy

class BinaryMulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    def backward(self, dout):
        dx = self.y * dout
        dy = self.x * dout
        return dx, dy

if __name__ == "__main__":
    x = 2.0
    y = 3.0
    z = 4.0
    # 建立節點
    addLayer = BinaryAddLayer()
    mulLayer = BinaryMulLayer()
    # 正向傳播
    h = addLayer.forward(x, y)
    z = mulLayer.forward(h, z)

    # 反向傳播
    d_loss = 1.0
    d_h, d_z = mulLayer.backward(d_loss)
    d_x, d_y = addLayer.backward(d_h)
    print(f"dx = {d_x}, dy = {d_y}, dz = {d_z}")


