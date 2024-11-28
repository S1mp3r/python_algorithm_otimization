import numpy as np
import matplotlib.pyplot as plt
import Local_Random_Search
import Hill_Climbing
import Global_Random_Search

def func(x1, x2):
    termo1 = x1 * np.cos(x1) / 20
    termo2 = 2 * np.exp((-x1**2) - (x2 - 1)**2)
    termo3 = 0.01 * x1 * x2
    
    return termo1 + termo2 + termo3

Hill_Climbing.hill_Climbing_init_(
                              rounds = 100,
                              limit_1 = -10,
                              limit_2 = 10,
                              limit_3 = -10,
                              limit_4 = 10,
                              f_apt = func,
                              minimization=False
                              )

Local_Random_Search.LRS_init_(
                              rounds = 100,
                              limit_1 = -10,
                              limit_2 = 10,
                              limit_3 = -10,
                              limit_4 = 10,
                              f_apt = func,
                              minimization=False
                              )

Global_Random_Search.GRS_init_(
                              rounds = 100,
                              limit_1 = -10,
                              limit_2 = 10,
                              limit_3 = -10,
                              limit_4 = 10,
                              f_apt = func,
                              minimization=False
                              )

bp = 1