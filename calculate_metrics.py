import numpy as np
from math import dist
def curve_rate(curve):
    # BC 弧曲率
    x = np.sqrt(np.diff(curve[:, 1]) ** 2 + np.diff(curve[:, 0]) ** 2)
    real_len = np.sum(x)
    straight_len = dist(curve[0], curve[-1])
    bc_length = real_len / straight_len

    return np.round(bc_length, 5), straight_len

def get_AUC(A, length):
    # formula
    # (length**2) / 4 / length
    return A + (length ** 2) / 4 / length
