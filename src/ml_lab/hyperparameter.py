import numpy as np

def log_scale_sample(low_exponent, high_exponent):
    """
    對數尺度隨機採樣器
    low_exponent: 指數下界 a (例如 -5 對應 10^-5)
    high_exponent: 指數上界 b (例如 -1 對應 10^-1)
    """
    # -------------------------------------------------------------------------
    # 實作對數採樣公式: 10^(Uniform(a, b))
    # -------------------------------------------------------------------------
    exponent = np.random.uniform(low_exponent, high_exponent)
    return 10 ** exponent

