import numpy as np
import matplotlib.pyplot as plt

def MQO(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y
