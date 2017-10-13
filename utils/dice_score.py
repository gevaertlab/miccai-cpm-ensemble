import numpy as np

def dice_score(x, y):
    x_bool = x.astype(bool)
    y_bool = y.astype(bool)
    num = np.sum(np.logical_and(x_bool, y_bool)) * 2.
    denom = np.sum(x_bool) + np.sum(y_bool)
    return num/denom
